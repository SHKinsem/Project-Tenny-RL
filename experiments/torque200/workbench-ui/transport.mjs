// One supervisor heartbeat and one auxiliary lane. Only /state establishes connectivity.
const defaultClock = {
  now: () => Date.now(),
  setTimeout: (callback, delay) => globalThis.setTimeout(callback, delay),
  clearTimeout: timer => globalThis.clearTimeout(timer),
};

export class APIError extends Error {
  constructor(message, { reason = 'network', details = '', status = null, cause } = {}) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'APIError'; this.reason = reason; this.details = details; this.status = status;
  }
}

const rawError = error => error ? `${error.name || 'Error'}: ${error.message || String(error)}` : '';
const friendlyError = reason => ({ timeout: '请求响应超时', network: '暂时无法连接服务', server: '服务暂时不可用' })[reason] || '暂时无法连接服务';

export async function apiJSON(path, { method = 'GET', body, fetchImpl = globalThis.fetch, timeoutMs = 8000, signal, clock = defaultClock } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timeout = clock.setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  const cancel = () => { clock.clearTimeout(timeout); controller.abort(); };
  signal?.addEventListener('abort', cancel, { once: true });
  if (signal?.aborted) cancel();
  try {
    let response;
    try {
      response = await fetchImpl(path, { method, cache: 'no-store', signal: controller.signal,
        headers: body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    } catch (error) {
      const reason = timedOut ? 'timeout' : 'network';
      throw new APIError(friendlyError(reason), { reason, details: `${method} ${path}${timedOut ? ` exceeded ${timeoutMs} ms` : ''}; ${rawError(error)}`, cause: error });
    }
    let data;
    try { data = await response.json(); }
    catch (error) {
      const reason = timedOut ? 'timeout' : 'server';
      throw new APIError(friendlyError(reason), { reason, status: response.status,
        details: `${method} ${path}; HTTP ${response.status}; invalid JSON; ${rawError(error)}`, cause: error });
    }
    if (!response.ok || data?.ok === false) {
      const message = (typeof data?.error === 'string' ? data.error : data?.error?.message) || data?.message || `HTTP ${response.status}`;
      throw new APIError(message, { reason: 'server', status: response.status, details: `${method} ${path}; HTTP ${response.status}; ${message}` });
    }
    return data;
  } finally {
    clock.clearTimeout(timeout);
    signal?.removeEventListener('abort', cancel);
  }
}

const labels = { dto: '累计记录', learning: '学习窗口', checkpoints: '模型记录', catalog: '模式目录', descriptor: '运行描述', capabilities: '操作能力', parameters: '参数定义', checkpointOptions: '续训模型', storage: '保存目录', localDirectories: '本机目录桥接', viewports: '预览窗口' };
const boundFields = ['dto', 'learning', 'checkpoints', 'descriptor', 'capabilities', 'parameters', 'checkpointOptions', 'viewports'];
const auxiliaryOrder = ['dto', 'viewports', 'parameters', 'capabilities', 'learning', 'checkpointOptions', 'checkpoints', 'descriptor', 'catalog', 'storage', 'localDirectories'];
const cadences = { dto: 1000, viewports: 1000, parameters: 15000, capabilities: 10000, learning: 10000,
  checkpointOptions: 30000, checkpoints: 30000, descriptor: 30000, catalog: 30000, storage: 30000, localDirectories: 30000 };

export function createWorkbenchTransport({ onSnapshot, fetchImpl = globalThis.fetch, intervalMs = 1000,
  maxBackoffMs = 8000, offlineAfter = 3, requestTimeoutMs = 8000, clock = defaultClock } = {}) {
  let stopped = true, lifecycle = 0, timer = null, heartbeat = null, auxiliary = null, followup = false;
  let sequence = 0, cacheKey = null, binding = 0, parameterCheckpoint = null, parameterContext = 0;
  let previousActive = null, previousSave = '', failures = 0;
  let readQueue = [], commands = [], waiters = [];
  const readAt = new Map(), versions = new Map(), warnings = new Map();
  let snapshot = { legacy: null, dto: null, learning: null, checkpoints: null, catalog: null, descriptor: null,
    capabilities: null, parameters: null, checkpointOptions: null, storage: null, localDirectories: null, viewports: null,
    connected: false, error: null, warnings: [], receivedAt: null,
    connection: { status: 'connecting', failures: 0, lastSuccessAt: null, reason: null, details: null, nextRetryAt: null } };
  const alive = token => !stopped && lifecycle === token;
  const emit = () => { if (!stopped) onSnapshot?.(snapshot); };
  const updateWarnings = () => { snapshot = { ...snapshot, warnings: [...warnings.values()] }; };
  const getVersion = name => versions.get(name) || 0;

  function invalidateRead(name) {
    readAt.delete(name); versions.set(name, getVersion(name) + 1);
    readQueue = readQueue.filter(job => job.name !== name);
  }
  function cancelRead() {
    if (auxiliary?.type === 'read') { auxiliary.cancelled = true; auxiliary.controller.abort(); }
  }
  function requestPath(name) {
    const mode = encodeURIComponent(snapshot.legacy?.mode || '');
    return {
      dto: `/api/workbench/v1/state?mode=${mode}`, viewports: '/api/workbench/v1/viewports',
      parameters: `/api/workbench/v1/parameters?mode=${mode}${parameterCheckpoint ? `&checkpoint=${encodeURIComponent(parameterCheckpoint)}` : ''}`,
      capabilities: `/api/workbench/v1/capabilities?mode=${mode}`, learning: '/learning-data',
      checkpointOptions: `/api/workbench/v1/checkpoint-options?mode=${mode}&limit=50`,
      checkpoints: `/api/workbench/v1/checkpoints?mode=${mode}&limit=50`, descriptor: `/api/workbench/v1/descriptor?mode=${mode}`,
      catalog: '/api/workbench/v1/catalog', storage: '/api/workbench/v1/storage', localDirectories: '/api/local/directories',
    }[name];
  }
  function queueRead(name, front = false) {
    if (!Object.hasOwn(cadences, name) || !snapshot.connected || !cacheKey) return;
    if (name === 'viewports' && Array.isArray(snapshot.legacy?.viewports)) return;
    if (readQueue.some(job => job.name === name)) return;
    if (auxiliary?.type === 'read' && !auxiliary.cancelled && auxiliary.binding === binding
        && auxiliary.name === name && auxiliary.version === getVersion(name)) return;
    const job = { type: 'read', name, binding, version: getVersion(name), context: parameterContext, path: requestPath(name) };
    if (front) readQueue.unshift(job); else readQueue.push(job);
  }
  function queueDue() {
    if (stopped) return;
    const now = clock.now();
    for (const name of auxiliaryOrder) if (!readAt.has(name) || now - readAt.get(name) >= cadences[name]) queueRead(name);
    drainAuxiliary();
  }
  function currentRead(job) {
    return alive(job.token) && !job.cancelled && job.binding === binding && job.version === getVersion(job.name)
      && (job.name !== 'parameters' || job.context === parameterContext);
  }
  function drainAuxiliary() {
    if (stopped || auxiliary) return;
    const job = commands.shift() || (snapshot.connected ? readQueue.shift() : null);
    if (!job) return;
    job.token = lifecycle; job.controller = new AbortController(); auxiliary = job;
    if (job.type === 'read') readAt.set(job.name, clock.now());
    const request = job.type === 'command' ? job.request : { path: job.path };
    apiJSON(request.path, { fetchImpl, clock, signal: job.controller.signal,
      timeoutMs: job.type === 'command' ? request.timeoutMs || 30000 : requestTimeoutMs,
      ...(job.type === 'command' ? { method: 'POST', body: request.body } : {}) }).then(result => {
      if (job.type === 'command') {
        if (alive(job.token)) job.resolve(result);
      } else if (currentRead(job)) {
        warnings.delete(job.name);
        snapshot = { ...snapshot, [job.name]: result }; updateWarnings(); emit();
      }
    }).catch(error => {
      if (job.type === 'command') {
        if (alive(job.token)) job.reject(error);
      } else if (currentRead(job)) {
        warnings.set(job.name, `${labels[job.name]}读取失败：${error.message}; ${error.details || rawError(error)}`);
        updateWarnings(); emit();
      }
    }).finally(() => {
      if (auxiliary === job) auxiliary = null;
      if (alive(job.token)) drainAuxiliary();
    });
  }

  function bindLegacy(legacy) {
    const id = legacy.training.run_id || String(legacy.training.run_dir || '').replace(/[\\/]+$/, '').split(/[\\/]/).at(-1) || '';
    const key = `${legacy.mode}/${id}`;
    if (key !== cacheKey) {
      cacheKey = key; binding++; parameterContext++;
      cancelRead(); readQueue = []; readAt.clear(); warnings.clear();
      snapshot = { ...snapshot, ...Object.fromEntries(boundFields.map(name => [name, null])) };
    }
    const save = JSON.stringify(legacy.training.workbench_save_results || []);
    if (legacy.training.active !== previousActive || save !== previousSave) {
      for (const name of ['parameters', 'checkpointOptions', 'checkpoints', 'storage']) invalidateRead(name);
    }
    previousActive = legacy.training.active; previousSave = save;
    const now = clock.now(); failures = 0;
    const embedded = Array.isArray(legacy.viewports)
      ? { viewports: { schemaVersion: 1, viewports: legacy.viewports, limit: 4 } } : {};
    if (embedded.viewports) {
      invalidateRead('viewports'); warnings.delete('viewports');
      if (auxiliary?.type === 'read' && auxiliary.name === 'viewports') cancelRead();
    }
    snapshot = { ...snapshot, ...embedded, legacy, connected: true, error: null, receivedAt: now,
      connection: { status: 'connected', failures: 0, lastSuccessAt: now, reason: null, details: null, nextRetryAt: null } };
    updateWarnings(); emit();
  }
  function retryDelay() { return Math.min(maxBackoffMs, intervalMs * (2 ** Math.min(30, Math.max(0, failures - 1)))); }
  function schedule(delay) {
    clock.clearTimeout(timer);
    if (!stopped) timer = clock.setTimeout(() => { timer = null; startHeartbeat(); }, delay);
  }
  function finishWaiters(readSequence) {
    const ready = waiters.filter(waiter => waiter.sequence <= readSequence);
    waiters = waiters.filter(waiter => waiter.sequence > readSequence);
    ready.forEach(waiter => waiter.resolve(snapshot));
  }
  function startHeartbeat() {
    if (stopped || heartbeat) return;
    clock.clearTimeout(timer); timer = null;
    const job = { token: lifecycle, sequence: ++sequence, controller: new AbortController() }; heartbeat = job;
    apiJSON('/state', { fetchImpl, timeoutMs: requestTimeoutMs, signal: job.controller.signal, clock }).then(legacy => {
      if (!alive(job.token)) return;
      if (!legacy || typeof legacy.mode !== 'string' || !legacy.training || typeof legacy.training !== 'object' || Array.isArray(legacy.training)) {
        throw new APIError('服务暂时不可用', { reason: 'server', details: 'GET /state: response is missing mode or training supervisor' });
      }
      bindLegacy(legacy); queueDue();
    }).catch(error => {
      if (!alive(job.token)) return;
      failures++;
      const reason = ['timeout', 'network', 'server'].includes(error.reason) ? error.reason : 'network';
      const nextRetryAt = clock.now() + retryDelay();
      snapshot = { ...snapshot, connected: false, error: friendlyError(reason),
        connection: { ...snapshot.connection, status: failures >= offlineAfter ? 'offline' : 'reconnecting',
          failures, reason, details: error.details || rawError(error), nextRetryAt } };
      readQueue = []; cancelRead(); emit();
    }).finally(() => {
      if (heartbeat === job) heartbeat = null;
      if (!alive(job.token)) return;
      finishWaiters(job.sequence);
      const delay = followup ? 0 : failures ? retryDelay() : intervalMs;
      followup = false; schedule(delay);
    });
  }
  function refresh() {
    if (stopped) return Promise.resolve(snapshot);
    const minimum = sequence + 1;
    const result = new Promise(resolve => waiters.push({ sequence: minimum, resolve }));
    clock.clearTimeout(timer); timer = null;
    if (heartbeat) followup = true; else startHeartbeat();
    return result;
  }
  return {
    start() {
      if (!stopped) return;
      stopped = false; lifecycle++; failures = 0;
      snapshot = { ...snapshot, connected: false, error: null,
        connection: { ...snapshot.connection, status: snapshot.connection.lastSuccessAt === null ? 'connecting' : 'reconnecting', failures: 0, reason: null, details: null, nextRetryAt: null } };
      emit(); startHeartbeat();
    },
    refresh,
    setParameterCheckpoint(path) {
      const next = path || null;
      if (next === parameterCheckpoint) return;
      parameterCheckpoint = next; parameterContext++; invalidateRead('parameters');
      snapshot = { ...snapshot, parameters: null }; emit();
      if (auxiliary?.type === 'read' && auxiliary.name === 'parameters') cancelRead();
      queueRead('parameters', true); drainAuxiliary(); refresh();
    },
    invalidate(names = []) {
      for (const name of names) { invalidateRead(name); queueRead(name, name === 'parameters'); }
      drainAuxiliary(); return refresh();
    },
    stop() {
      if (stopped) return;
      stopped = true; lifecycle++; clock.clearTimeout(timer); timer = null; followup = false;
      const oldHeartbeat = heartbeat, oldAuxiliary = auxiliary;
      heartbeat = auxiliary = null;
      oldHeartbeat?.controller.abort(); oldAuxiliary?.controller.abort();
      const error = new APIError('页面已停止请求', { reason: 'network', details: 'transport stopped' });
      if (oldAuxiliary?.type === 'command') oldAuxiliary.reject(error);
      commands.splice(0).forEach(job => job.reject(error)); readQueue = [];
      waiters.splice(0).forEach(waiter => waiter.resolve(snapshot));
    },
    async command(request) {
      if (stopped) throw new APIError('页面已停止请求', { reason: 'network', details: 'transport stopped' });
      // Mutations use the auxiliary slot too, so four MJPEG streams still leave
      // a connection for the heartbeat. A POST is attempted exactly once.
      const result = await new Promise((resolve, reject) => {
        commands.push({ type: 'command', request, resolve, reject }); cancelRead(); drainAuxiliary();
      });
      if (['/api/workbench/v1/storage', '/api/workbench/v1/import'].includes(request.path)) {
        for (const name of ['storage', 'checkpointOptions', 'checkpoints', 'parameters']) invalidateRead(name);
      }
      // refresh always requires a heartbeat begun after this call, never an
      // earlier in-flight read that predates the accepted mutation.
      await refresh(); return result;
    },
  };
}
