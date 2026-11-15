import test from 'node:test';
import assert from 'node:assert/strict';
import { APIError, apiJSON, createWorkbenchTransport } from './transport.mjs';

const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const flush = async () => { for (let turn = 0; turn < 200; turn++) await Promise.resolve(); };
const response = (data, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => data });
const legacy = (steps = 1, extra = {}) => ({ mode: 'leg_force_bc', training: { run_id: 'run-a', active: true, phase: 'training', steps }, viewports: [{ id: '1', suspended: true }], ...extra });

class Clock {
  value = 0; nextId = 0; timers = new Map();
  now = () => this.value;
  setTimeout = (callback, delay) => { const id = ++this.nextId; this.timers.set(id, { at: this.value + delay, callback }); return id; };
  clearTimeout = id => { this.timers.delete(id); };
  async tick(delay) {
    const target = this.value + delay;
    for (let count = 0; count < 2000; count++) {
      const next = [...this.timers.entries()].filter(([, timer]) => timer.at <= target).sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
      if (!next) { this.value = target; await flush(); return; }
      this.value = next[1].at; this.timers.delete(next[0]); next[1].callback(); await flush();
    }
    throw new Error('Clock did not quiesce');
  }
}

function abortable(signal) {
  return new Promise((_, reject) => {
    const abort = () => reject(Object.assign(new Error('signal is aborted without reason'), { name: 'AbortError' }));
    if (signal.aborted) abort(); else signal.addEventListener('abort', abort, { once: true });
  });
}

test('a slow auxiliary read never delays the next supervisor heartbeat; one read per lane', async () => {
  const clock = new Clock(), held = deferred(), events = [], calls = [];
  let reads = 0, active = 0, max = 0;
  const fetchImpl = async (path, options) => {
    calls.push(path); active++; max = Math.max(max, active);
    try {
      if (path === '/state') return response(legacy(++reads));
      if (path.startsWith('/api/workbench/v1/state?')) await Promise.race([held.promise, abortable(options.signal)]);
      return response({});
    } finally { active--; }
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => events.push(value) });
  try {
    transport.start(); await flush();
    assert.equal(events.at(-1).legacy.training.steps, 1);
    assert.equal(events.at(-1).connected, true);
    assert.equal(active, 1);
    await clock.tick(1000);
    assert.equal(events.at(-1).legacy.training.steps, 2);
    assert.equal(active, 1); assert.equal(max, 2);
    assert.equal(calls.filter(path => path !== '/state').length, 1);
    assert.equal(calls.includes('/api/workbench/v1/viewports'), false);
    assert.deepEqual(events.at(-1).viewports, { schemaVersion: 1, viewports: [{ id: '1', suspended: true }], limit: 4 });
    held.resolve(); await flush();
    assert.ok(calls.some(path => path.startsWith('/api/workbench/v1/parameters?')));
  } finally { transport.stop(); await flush(); }
  assert.equal(clock.timers.size, 0);
});

test('four long MJPEG connections leave room for a slow auxiliary read and fresh state', async () => {
  const clock = new Clock(), streamEnd = deferred(), auxiliaryEnd = deferred(), snapshots = [];
  let open = 0, maximum = 0, stateReads = 0, queued = 0;
  const waiters = [];
  const acquire = () => open < 6 ? (open++, maximum = Math.max(maximum, open), Promise.resolve()) : new Promise(resolve => { queued++; waiters.push(resolve); });
  const release = () => { open--; if (waiters.length) { open++; maximum = Math.max(maximum, open); waiters.shift()(); } };
  const fetchImpl = async (path, options = {}) => {
    await acquire();
    try {
      if (path.startsWith('/stream.mjpg')) { await streamEnd.promise; return response({}); }
      if (path === '/state') return response(legacy(++stateReads));
      if (path.startsWith('/api/workbench/v1/state?')) await Promise.race([auxiliaryEnd.promise, abortable(options.signal)]);
      return response({});
    } finally { release(); }
  };
  const streams = ['1', '2', '3', '4'].map(id => fetchImpl(`/stream.mjpg?viewport=${id}`));
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => snapshots.push(value) });
  try {
    transport.start(); await flush();
    await clock.tick(3000);
    assert.equal(stateReads, 4); assert.equal(snapshots.at(-1).connected, true);
    assert.equal(maximum, 6); assert.equal(queued, 0);
  } finally {
    transport.stop(); streamEnd.resolve(); auxiliaryEnd.resolve(); await Promise.all(streams); await flush();
  }
  assert.equal(open, 0); assert.equal(clock.timers.size, 0);
});

test('auxiliary failures preserve data and only add field-specific technical warnings', async () => {
  const clock = new Clock(); let received, fail = false;
  const fetchImpl = async path => {
    if (path === '/state') return response(legacy());
    if (path === '/learning-data') {
      if (fail) throw new TypeError('ECONNRESET fixture');
      return response({ points: [{ steps: 1, value_loss: 0.5 }] });
    }
    return response({});
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => { received = value; } });
  try {
    transport.start(); await flush(); const previous = received.learning;
    fail = true; const refreshed = transport.invalidate(['learning']); await flush(); await refreshed;
    assert.equal(received.connected, true); assert.equal(received.connection.status, 'connected');
    assert.equal(received.learning, previous); assert.equal(received.error, null);
    assert.match(received.warnings.join(' '), /学习窗口.*ECONNRESET/);
  } finally { transport.stop(); await flush(); }
});

test('a first failed heartbeat disables writes, consecutive failures back off, recovery resets them', async () => {
  const clock = new Clock(), snapshots = [];
  let reads = 0;
  const fetchImpl = async path => {
    if (path !== '/state') return response({});
    reads++;
    if (reads > 1 && reads < 6) throw new TypeError('net::ERR_CONNECTION_RESET fixture');
    return response(legacy(reads));
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => snapshots.push(value) });
  try {
    transport.start(); await flush();
    assert.equal(snapshots[0].connection.status, 'connecting');
    await clock.tick(1000);
    let current = snapshots.at(-1);
    assert.equal(current.connected, false); assert.equal(current.connection.status, 'reconnecting');
    assert.equal(current.connection.failures, 1); assert.equal(current.connection.reason, 'network');
    assert.equal(current.error, '暂时无法连接服务'); assert.match(current.connection.details, /ERR_CONNECTION_RESET/);
    assert.equal(current.legacy.training.steps, 1); assert.equal(current.connection.lastSuccessAt, 0);
    assert.equal(current.connection.nextRetryAt, 2000);
    await clock.tick(999); assert.equal(reads, 2);
    await clock.tick(1); assert.equal(snapshots.at(-1).connection.nextRetryAt, 4000);
    await clock.tick(2000); current = snapshots.at(-1);
    assert.equal(current.connection.status, 'offline'); assert.equal(current.connection.failures, 3);
    assert.equal(current.connection.nextRetryAt, 8000);
    await clock.tick(4000); assert.equal(snapshots.at(-1).connection.nextRetryAt, 16000);
    await clock.tick(8000); current = snapshots.at(-1);
    assert.equal(reads, 6); assert.equal(current.connected, true); assert.equal(current.connection.status, 'connected');
    assert.equal(current.connection.failures, 0); assert.equal(current.connection.reason, null); assert.equal(current.connection.details, null);
    assert.equal(current.connection.lastSuccessAt, 16000); assert.equal(current.connection.nextRetryAt, null);
  } finally { transport.stop(); await flush(); }
});

test('apiJSON classifies timeout, network and server errors without using AbortError as the message', async () => {
  const clock = new Clock();
  const pending = apiJSON('/slow', { clock, timeoutMs: 100, fetchImpl: async (_, options) => abortable(options.signal) });
  const failed = assert.rejects(pending, error => {
    assert.ok(error instanceof APIError); assert.equal(error.reason, 'timeout'); assert.equal(error.message, '请求响应超时');
    assert.match(error.details, /GET \/slow exceeded 100 ms/); return true;
  });
  await clock.tick(100); await failed;
  await assert.rejects(apiJSON('/network', { clock, fetchImpl: async () => { throw new TypeError('DNS fixture'); } }), error => {
    assert.equal(error.reason, 'network'); assert.equal(error.message, '暂时无法连接服务'); assert.match(error.details, /DNS fixture/); return true;
  });
  await assert.rejects(apiJSON('/server', { clock, fetchImpl: async () => response({ error: { message: '模型暂不可用' } }, 503) }), error => {
    assert.equal(error.reason, 'server'); assert.equal(error.status, 503); assert.equal(error.message, '模型暂不可用'); return true;
  });
  assert.equal(clock.timers.size, 0);
});

test('invalid supervisor JSON is a server failure and never authorizes connectivity', async () => {
  const clock = new Clock(); let received;
  const transport = createWorkbenchTransport({ clock, fetchImpl: async () => response({ mode: 'leg_force_bc', training: [] }), onSnapshot: value => { received = value; } });
  try {
    transport.start(); await flush();
    assert.equal(received.connected, false); assert.equal(received.connection.reason, 'server');
    assert.match(received.connection.details, /missing mode or training supervisor/);
  } finally { transport.stop(); await flush(); }
});

test('a command waits for a state request begun after the accepted POST, never a pre-command read', async () => {
  const clock = new Clock(), firstRead = deferred(), calls = []; let stateReads = 0, received, done = false;
  const fetchImpl = async (path, options) => {
    if (options.method === 'POST') { calls.push('POST'); return response({ ok: true, requestId: 'one' }); }
    if (path === '/state') {
      const count = ++stateReads; calls.push(`state-${count}-begin`);
      if (count === 1) await firstRead.promise;
      calls.push(`state-${count}-end`); return response(legacy(count));
    }
    return response({});
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => { received = value; } });
  try {
    transport.start(); await flush();
    const submitted = transport.command({ path: '/control', body: { action: 'reset' } }).then(value => { done = true; return value; });
    await flush(); assert.equal(calls.filter(value => value === 'POST').length, 1);
    firstRead.resolve(); await flush();
    assert.equal(stateReads, 1); assert.equal(done, false);
    await clock.tick(0); const result = await submitted;
    assert.equal(result.requestId, 'one'); assert.equal(received.legacy.training.steps, 2);
    assert.deepEqual(calls.filter(value => value.startsWith('state') || value === 'POST'), ['state-1-begin', 'POST', 'state-1-end', 'state-2-begin', 'state-2-end']);
  } finally { transport.stop(); await flush(); }
});

test('failed mutations are not retried by automatic supervisor polling', async () => {
  const clock = new Clock(); let posts = 0;
  const fetchImpl = async (path, options) => {
    if (options.method === 'POST') { posts++; return response({ error: 'write failed fixture' }, 503); }
    return response(path === '/state' ? legacy() : {});
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl });
  try {
    transport.start(); await flush();
    await assert.rejects(transport.command({ path: '/training', body: { action: 'save' } }), /write failed fixture/);
    await clock.tick(3000); assert.equal(posts, 1);
  } finally { transport.stop(); await flush(); }
});

test('a stale parameter schema cannot overwrite the new checkpoint context even if fetch ignores abort', async () => {
  const clock = new Clock(), stale = deferred(), schemas = [];
  let entered = false, reads = 0, current;
  const fetchImpl = async path => {
    if (path === '/state') return response(legacy());
    if (path.startsWith('/api/workbench/v1/parameters?')) {
      const checkpoint = new URL(path, 'http://fixture.invalid').searchParams.get('checkpoint');
      if (++reads === 1) { entered = true; await stale.promise; }
      return response({ mode: 'leg_force_bc', checkpoint, fields: [] });
    }
    return response({});
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => { current = value; if (value.parameters) schemas.push(value.parameters.checkpoint); } });
  try {
    transport.start(); await flush(); assert.equal(entered, true);
    transport.setParameterCheckpoint('/models/B'); await flush();
    assert.equal(current.parameters, null);
    stale.resolve(); await flush();
    assert.equal(current.parameters.checkpoint, '/models/B');
    assert.deepEqual([...new Set(schemas)], ['/models/B']);
  } finally { transport.stop(); await flush(); }
});

test('mode and run changes clear bound values and reject a late response from the previous run', async () => {
  const clock = new Clock(), oldDTO = deferred(), snapshots = []; let reads = 0, firstDTO = true;
  const fetchImpl = async path => {
    if (path === '/state') return response(++reads === 1 ? legacy() : legacy(2, { mode: 'torque200', training: { run_id: 'run-b', active: true, phase: 'training', steps: 2 } }));
    if (path.startsWith('/api/workbench/v1/state?')) {
      const old = firstDTO; firstDTO = false;
      if (old) await oldDTO.promise;
      return response({ mode: old ? 'leg_force_bc' : 'torque200', run: { id: old ? 'run-a' : 'run-b' } });
    }
    return response({});
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => snapshots.push(value) });
  try {
    transport.start(); await flush(); const refreshed = transport.refresh(); await flush(); await refreshed;
    assert.equal(snapshots.at(-1).legacy.mode, 'torque200'); assert.equal(snapshots.at(-1).dto, null);
    oldDTO.resolve(); await flush();
    assert.equal(snapshots.at(-1).dto.mode, 'torque200');
    assert.ok(snapshots.filter(value => value.legacy?.mode === 'torque200').every(value => value.dto === null || value.dto.mode === 'torque200'));
  } finally { transport.stop(); await flush(); }
});

test('stop cancels timers and ignores late state or auxiliary responses without callbacks', async () => {
  for (const holdState of [true, false]) {
    const clock = new Clock(), held = deferred(), snapshots = []; let calls = 0;
    const fetchImpl = async path => {
      calls++;
      if ((path === '/state') === holdState) await held.promise;
      return response(path === '/state' ? legacy() : {});
    };
    const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => snapshots.push(value) });
    transport.start(); await flush(); const before = snapshots.length, callsBefore = calls;
    transport.stop(); assert.equal(clock.timers.size, 0);
    held.resolve(); await flush(); await clock.tick(60000);
    assert.equal(snapshots.length, before); assert.equal(calls, callsBefore); assert.equal(clock.timers.size, 0);
  }
});

test('transport can restart immediately while an aborted old request is still settling', async () => {
  const clock = new Clock(), old = deferred(), snapshots = []; let reads = 0;
  const fetchImpl = async path => {
    if (path === '/state') { const count = ++reads; if (count === 1) await old.promise; return response(legacy(count)); }
    return response({});
  };
  const transport = createWorkbenchTransport({ clock, fetchImpl, onSnapshot: value => snapshots.push(value) });
  try {
    transport.start(); transport.stop(); transport.start(); await flush();
    assert.equal(snapshots.at(-1).legacy.training.steps, 2);
    old.resolve(); await flush(); assert.equal(snapshots.at(-1).legacy.training.steps, 2);
  } finally { transport.stop(); await flush(); }
});

test('auxiliary reads follow their 10/15/30-second cadence and old servers get a viewport fallback', async () => {
  const clock = new Clock(), calls = [];
  const fetchImpl = async path => { calls.push(path); return response(path === '/state' ? legacy(1, { viewports: undefined }) : {}); };
  const count = part => calls.filter(path => path.includes(part)).length;
  const transport = createWorkbenchTransport({ clock, fetchImpl });
  try {
    transport.start(); await flush();
    assert.deepEqual(calls.slice(1, 4).map(path => path.split('?')[0]), ['/api/workbench/v1/state', '/api/workbench/v1/viewports', '/api/workbench/v1/parameters']);
    assert.equal(count('/learning-data'), 1); assert.equal(count('/parameters?'), 1); assert.equal(count('/catalog'), 1);
    await clock.tick(10000);
    assert.equal(count('/learning-data'), 2); assert.equal(count('/parameters?'), 1); assert.equal(count('/catalog'), 1);
    await clock.tick(5000); assert.equal(count('/parameters?'), 2);
    await clock.tick(15000);
    assert.equal(count('/learning-data'), 4); assert.equal(count('/parameters?'), 3); assert.equal(count('/catalog'), 2);
  } finally { transport.stop(); await flush(); }
});
