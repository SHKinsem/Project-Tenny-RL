// Pure boundary between reported server data and the workbench. No clocks or I/O.
const finite = value => typeof value === 'number' && Number.isFinite(value);
const number = value => finite(value) ? value : null;
const nonnegative = value => finite(value) && value >= 0 ? value : null;
const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
export const basename = value => typeof value === 'string' ? value.replace(/[\\/]+$/, '').split(/[\\/]/).at(-1) || '' : '';
export function runId(training) { return typeof training?.run_id === 'string' && training.run_id ? training.run_id : basename(training?.run_dir); }
export const runKey = state => `${state?.mode || ''}/${runId(state?.training)}`;
export function normalizeCatalog(catalog) {
  return Array.isArray(catalog?.modes) ? catalog.modes.filter(item => typeof item?.id === 'string' && item.id).map(item => ({ ...item, label: typeof item.label === 'string' && item.label ? item.label : item.id })) : [];
}
export const modeLabel = (mode, catalog) => normalizeCatalog(catalog).find(item => item.id === mode)?.label || mode || '未报告';
export function formatNumber(value) { return finite(value) ? value.toLocaleString('zh-CN', { maximumFractionDigits: 0 }) : '—'; }
export function formatSeconds(value) {
  if (!finite(value)) return '未知';
  const seconds = Math.max(0, Math.floor(value));
  return seconds >= 3600 ? `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}` : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}
export function producerTime(value) {
  if (finite(value) && value > 0) return value < 1e12 ? value * 1000 : value;
  if (typeof value === 'string' && value) { const parsed = Date.parse(value); return finite(parsed) ? parsed : null; }
  return null;
}
function firstNumber(...values) { return values.find(finite) ?? null; }
function reportedConfig(training, descriptor = null) {
  const status = object(training.status);
  const runtime = object(training.runtime_profile || status.runtime_profile);
  const effective = object(runtime.effective_settings || runtime.settings);
  const saved = object(descriptor?.runtime);
  const savedKeys = { num_envs: 'numEnvs', batch_size: 'batchSize', num_minibatches: 'numMinibatches', unroll_length: 'unrollLength', num_updates_per_batch: 'numUpdatesPerBatch', learning_rate: 'learningRate', num_eval_envs: 'numEvalEnvs', discounting: 'discounting', gae_lambda: 'gaeLambda', clipping_epsilon: 'clippingEpsilon' };
  const savedConfig = Object.fromEntries(Object.entries(savedKeys).map(([key, savedKey]) => [key, saved[savedKey]]));
  const sources = [effective, status, training, savedConfig];
  const get = key => firstNumber(...sources.map(source => source[key]));
  return {
    runtime: Object.keys(runtime).length ? runtime : null,
    num_envs: get('num_envs'), batch_size: get('batch_size'), num_minibatches: get('num_minibatches'),
    unroll_length: get('unroll_length'), num_updates_per_batch: get('num_updates_per_batch'),
    learning_rate: get('learning_rate'), discounting: get('discounting'), gae_lambda: get('gae_lambda'),
    clipping_epsilon: get('clipping_epsilon'), seed: get('seed'), num_eval_envs: get('num_eval_envs'),
    eval_interval_seconds: get('eval_interval_seconds'), episode_length: get('episode_length'),
  };
}
function canonicalCheckpointId(value) {
  return typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value) && value !== '00000000-0000-0000-0000-000000000000' ? value : null;
}
function capturedCheckpointSteps(checkpoint, steps) {
  if (finite(steps)) return steps;
  const tail = basename(checkpoint);
  const match = tail.match(/(?:checkpoint|step|params|model)[_-]?(\d+)(?:\.[^.]+)?$/i) || tail.match(/^(\d+)(?:\.[^.]+)?$/);
  return match ? number(Number(match[1])) : null;
}
export function checkpointLabel(checkpoint, steps = null, alias = null, checkpointId = null) {
  const identity = canonicalCheckpointId(checkpointId);
  if (identity) return `CP·${identity.slice(0, 8).toUpperCase()}`;
  if (typeof alias === 'string' && alias.trim()) return alias.trim();
  const captured = capturedCheckpointSteps(checkpoint, steps);
  return captured !== null ? `${formatNumber(captured)} 步` : basename(checkpoint) || '未报告';
}
export function checkpointTitle(checkpoint, steps = null, checkpointId = null) {
  const identity = canonicalCheckpointId(checkpointId), captured = capturedCheckpointSteps(checkpoint, steps);
  return [identity && `checkpoint UUID：${identity}`, captured !== null && `本轮捕获步数：${formatNumber(captured)}`, checkpoint && `路径：${checkpoint}`].filter(Boolean).join('\n');
}
const METRIC_KEYS = { episode_seconds: 'episodeSeconds', episode_reward: 'reward', value_loss: 'valueLoss', policy_loss: 'policyLoss' };
const metricKey = key => METRIC_KEYS[key] || key;
/** Exclude unsampled events, while retaining explicit invalid measurements and gaps. */
export function metricRecords(points, key) {
  return Array.isArray(points) ? points.filter(point => !Array.isArray(point?.measured) || point.measured.includes(key) || point.gap === true || Array.isArray(point.gap) && point.gap.includes(key)) : [];
}
const LOADING_LABELS = { connecting: '连接训练服务', syncing: '同步模型', loading_policy: '加载策略', loading_environment: '加载环境', first_frame: '等待首帧', preparing: '准备训练', compiling: '编译训练程序', teacher_collect: '采集教师数据', teacher_fit: '拟合初始策略', teacher_verify: '验证初始策略', teacher_gate: '检查训练条件' };
/** Only reported progress is determinate. Unknown work never receives a made-up percentage. */
export function loadingView(source = {}, fallback = '准备中') {
  source = object(source);
  const error = typeof source.error === 'string' ? source.error : source.error?.message || (source.stage === 'error' ? source.message || '加载失败' : null);
  const active = source.active === true && !error && source.stage !== 'error';
  const total = number(source.total), completed = number(source.completed ?? source.current);
  const reported = number(source.progress);
  const percent = total > 0 && completed !== null ? Math.max(0, Math.min(100, completed / total * 100)) : reported !== null && reported >= 0 && reported <= 100 ? reported : null;
  return { active, visible: active || Boolean(error) || source.stage === 'error', stage: source.stage || null, label: error || source.message || LOADING_LABELS[source.stage] || fallback, percent, indeterminate: active && percent === null, error };
}
/** Image failures belong to one attached request, including requests for the first frame. */
export function previewImageFailure(request, current) {
  if (!request?.key || request.key !== current?.key || request.generation !== current.generation || current.open !== true || current.intendedOpen === false) return null;
  return loadingView({ stage: 'error', error: '预览图像请求失败，请重试。' });
}
export function connectionView(connection, { connected = false, known = false, error = null, receivedAt = null } = {}) {
  const source = object(connection), valid = ['connecting', 'connected', 'reconnecting', 'offline'];
  let status = valid.includes(source.status) ? source.status : connected && known ? 'connected' : known || error ? 'offline' : 'connecting';
  if (status === 'connected' && (!connected || !known)) status = known ? 'reconnecting' : 'connecting';
  const label = { connecting: '正在连接', connected: '已连接', reconnecting: '正在重连', offline: '连接已断开' }[status];
  const detail = status === 'connected' ? '状态已更新；训练操作仍按服务端能力开放。' : known ? '保留最近一次报告，训练操作暂不可用；页面会自动重试。' : '正在获取服务端状态；尚未开放训练操作。';
  return { status, label, detail, fresh: Boolean(connected && known && status === 'connected'), failures: nonnegative(source.failures) ?? 0, lastSuccessAt: number(source.lastSuccessAt ?? receivedAt), nextRetryAt: number(source.nextRetryAt), reason: source.reason || null, technicalReason: typeof source.details === 'string' ? source.details : typeof error === 'string' ? error : error?.message || null, allowLocalPreviewClose: Boolean(known && (!connected || status !== 'connected')) };
}
export function previewRequestMode(action, view) {
  if (view.connected) return 'remote';
  return view.known && action === 'suspend' ? 'local-close' : 'blocked';
}
/** Reconnection can retain an existing request, but cannot create or change its version. */
export function previewStreamPlan(request, current) {
  if (!request?.key || current?.known !== true || current.open !== true || current.intendedOpen === false || request.mode !== current.mode || request.runId !== current.runId || request.generation !== current.generation) return { kind: 'close', key: null };
  return { kind: request.failed ? 'failed' : 'retain', key: request.key };
}
export function validateParameterDraft(fields, draft = {}) {
  const values = {}, errors = [];
  for (const [key, raw] of Object.entries(draft)) {
    const field = Array.isArray(fields) ? fields.find(item => item.key === key) : null;
    if (!field || field.editable !== true) { errors.push(`${field?.label || key} 当前不可编辑`); continue; }
    const value = field.type === 'boolean' ? raw : typeof raw === 'string' && raw.trim() === '' ? NaN : Number(raw);
    if (field.type === 'boolean' ? typeof value !== 'boolean' : !finite(value) || field.type === 'integer' && !Number.isInteger(value)) { errors.push(`${field.label || key} 需要${field.type === 'integer' ? '整数' : field.type === 'boolean' ? '开关值' : '有效数值'}`); continue; }
    if (field.type !== 'boolean' && (finite(field.min) && value < field.min || finite(field.max) && value > field.max)) { errors.push(`${field.label || key} 超出允许范围`); continue; }
    values[key] = value;
  }
  return { values, errors, valid: errors.length === 0 };
}
function controlCapability(capabilities, key) {
  const value = capabilities?.legacyControls?.[key];
  return value === true || value?.available === true || value?.enabled === true;
}
export function normalizeLearning(data, legacy, catalog = null) {
  const metadata = object(data?.metadata), currentId = runId(legacy?.training), identity = basename(metadata.run);
  const embeddedMode = typeof metadata.run === 'string' ? metadata.run.split(/[\\/]/).find(part => normalizeCatalog(catalog).some(item => item.id === part)) : null;
  const matches = Boolean(currentId && identity === currentId && (!metadata.mode || metadata.mode === legacy?.mode) && (!embeddedMode || embeddedMode === legacy?.mode));
  const reason = !data?.available ? metadata.message || '当前没有学习记录' : !matches ? '学习窗口与当前运行身份不一致，未合并显示' : '';
  const points = matches && data?.available && Array.isArray(data.points) ? data.points.map(point => ({
    steps: nonnegative(point?.steps),
    at: producerTime(point?.at ?? point?.recorded_at ?? point?.timestamp),
    episodeSeconds: number(point?.episode_seconds), reward: number(point?.episode_reward),
    valueLoss: number(point?.value_loss), policyLoss: number(point?.policy_loss),
    ...(Array.isArray(point?.measured) ? { measured: point.measured.map(metricKey) } : {}),
    ...(point?.gap === true ? { gap: true } : Array.isArray(point?.gap) ? { gap: point.gap.map(metricKey) } : {}),
  })).filter(point => point.steps !== null) : [];
  return { points, metadata, matches, reason, identity: metadata.run || null, last: points.at(-1) || null };
}
function progressView(training, dto, matches) {
  const status = object(training.status), source = matches ? object(dto.progress || dto.run) : {};
  const preparing = training.active === true && (training.training_clock_started ?? status.training_clock_started) === false;
  const elapsed = preparing ? 0 : matches ? nonnegative(dto.run?.elapsedSeconds) : nonnegative(training.elapsed_seconds ?? status.elapsed_seconds);
  const duration = matches ? nonnegative(dto.run?.durationSeconds) : nonnegative(training.duration_seconds ?? status.duration_seconds);
  const baseSeconds = nonnegative(source.baseTrainedSeconds), cumulativeSeconds = nonnegative(source.trainedSeconds);
  const cumulativeTargetSeconds = nonnegative(source.targetTrainedSeconds);
  const cumulativePercent = number(source.cumulativePercent);
  const segmentPercent = matches && finite(source.segmentPercent) ? source.segmentPercent : duration > 0 && elapsed !== null ? Math.min(100, elapsed / duration * 100) : null;
  const cumulative = cumulativeSeconds !== null;
  return {
    elapsedSeconds: elapsed, durationSeconds: duration, baseSeconds,
    trainedSeconds: cumulativeSeconds, targetSeconds: cumulativeTargetSeconds,
    trainedSteps: nonnegative(source.trainedSteps), segmentSteps: nonnegative(training.steps ?? status.steps),
    cumulativePercent, segmentPercent, percent: cumulativePercent ?? segmentPercent,
    clock: cumulative ? `${formatSeconds(cumulativeSeconds)} / ${cumulativeTargetSeconds === null ? '不限时 / 未知计划' : formatSeconds(cumulativeTargetSeconds)}` : `${formatSeconds(elapsed)} / ${duration === 0 ? '不限时' : formatSeconds(duration)}`,
    label: cumulative ? '累计 / 计划时长' : '本轮 / 预算时长',
    detail: cumulative ? `历史 ${formatSeconds(baseSeconds)}；本轮 ${formatSeconds(elapsed)}。计时以服务端报告为准。` : `历史累计未报告；本轮 ${formatSeconds(elapsed)}。计时以服务端报告为准。`,
    preparing, preparationSeconds: nonnegative(training.preparation_seconds ?? status.preparation_seconds),
  };
}
const PHASE_LABELS = { idle: '待开始', preparing: '准备中', compiling: '编译中', training: '训练中', running: '训练中', pausing: '等待安全边界暂停', paused: '训练已暂停', resuming: '正在继续训练', saving: '保存中', exporting: '导出中', verifying: '核验中', stopping: '正在停止并保存', completed: '已完成', stopped: '已停止', failed: '失败', cancelled: '已取消', interrupted: '已中断' };
export function buildWorkbenchView({ legacy = null, dto = null, learning = null, checkpoints = null, catalog = null, descriptor = null, capabilities = null, parameters = null, checkpointOptions = null, storage = null, localDirectories = null, viewports = null, connected = false, connection = null, receivedAt = null, error = null, warnings = [] } = {}) {
  const known = Boolean(legacy && typeof legacy.mode === 'string'), training = object(legacy?.training), latest = object(training.latest), id = runId(training);
  const connectivity = connectionView(connection, { connected, known, error, receivedAt });
  const supervisorKnown = typeof training.active === 'boolean';
  const writable = connectivity.fresh && supervisorKnown && legacy.switching !== true;
  const active = training.active === true;
  const dtoMatches = Boolean(dto?.available && dto.mode === legacy?.mode && dto.run?.id === id && id);
  const descriptorMatches = Boolean(descriptor?.available && descriptor.mode === legacy?.mode && descriptor.runId === id && id);
  const matchedDescriptor = descriptorMatches ? descriptor : null;
  const progress = progressView(training, dto, dtoMatches);
  const protocol = training.workbench_control_protocol === 1 || training.status?.workbench_control_protocol === 1;
  const controlResult = object(training.workbench_control_result), paused = training.paused === true, pauseRequested = training.pause_requested === true;
  const resuming = controlResult.action === 'resume' && controlResult.status === 'pending';
  const phase = !known ? 'unknown' : active && resuming ? 'resuming' : active && pauseRequested ? 'pausing' : active && paused ? 'paused' : progress.preparing ? 'preparing' : training.phase || (active ? 'training' : 'idle');
  const finishing = ['stopping', 'saving', 'exporting', 'verifying'].includes(phase);
  const saveResults = Array.isArray(training.workbench_save_results) ? training.workbench_save_results : [];
  const save = saveResults.at(-1) || null, savePending = saveResults.some(item => ['pending', 'saving'].includes(item.status));
  const latestCheckpoint = typeof training.latest_checkpoint === 'string' ? training.latest_checkpoint : latest.checkpoint || latest.export_path || null;
  const canResume = writable && !active && controlCapability(capabilities, 'resumeLatest') && training.can_resume === true && Boolean(latestCheckpoint);
  const checkpoint = latestCheckpoint || legacy?.checkpoint || null;
  const matchingCandidate = Array.isArray(checkpoints?.items) ? checkpoints.items.find(item => item.relativePath && checkpoint && checkpoint.replaceAll('\\', '/').endsWith(item.relativePath)) : null;
  const latestSteps = firstNumber(latest.actual_timesteps, latest.steps, latest.captured_steps, latest.capturedSteps, matchingCandidate?.capturedSteps);
  const checkpointId = canonicalCheckpointId(latest.workbench_checkpoint_id) || canonicalCheckpointId(matchingCandidate?.id);
  const alias = matchingCandidate?.alias || latest.alias || null;
  const curve = normalizeLearning(learning, legacy, catalog), modes = normalizeCatalog(catalog), mode = modes.find(item => item.id === legacy?.mode);
  const parameterFields = parameters?.mode === legacy?.mode && Array.isArray(parameters?.fields) ? parameters.fields : [];
  const options = checkpointOptions?.mode === legacy?.mode && Array.isArray(checkpointOptions?.checkpoints) ? checkpointOptions.checkpoints : [];
  const reportedActivity = !known ? loadingView({ active: !error, stage: 'connecting', error }, '连接训练服务') : progress.preparing ? loadingView({ ...object(training.preparation_progress || training.status?.preparation_progress), active: true, stage: training.stage || training.status?.stage || 'preparing', message: training.preparation_message || training.status?.preparation_message }, '准备训练环境') : loadingView();
  const activity = !known || connectivity.fresh ? reportedActivity : { ...reportedActivity, active: false, visible: false, indeterminate: false };
  const trainingPhaseLabel = PHASE_LABELS[phase] || (phase === 'unknown' ? '状态未知' : phase);
  return {
    known, connected: connectivity.fresh, connection: connectivity, error, mode: legacy?.mode || null, modes, modeLabel: modeLabel(legacy?.mode, catalog), runId: id || null, descriptor: matchedDescriptor, descriptorMatches,
    active, supervisorKnown, phase, trainingPhaseLabel, phaseLabel: connection ? trainingPhaseLabel : connected ? PHASE_LABELS[phase] || phase : known ? '连接中断 · 保留最后报告' : '正在连接',
    training, latest, checkpoint, checkpointId, checkpointLabel: checkpointLabel(checkpoint, latestSteps, alias, checkpointId), checkpointTitle: checkpointTitle(checkpoint, latestSteps, checkpointId), latestSteps,
    dtoMatches, progress, config: reportedConfig(training, matchedDescriptor), curve, activity, protocol, paused, pauseRequested, controlResult, saveResults, save, savePending,
    parameterFields, parameterSchema: parameters, options, storage, localDirectories, viewports: Array.isArray(viewports?.viewports) ? viewports.viewports : [],
    controls: { start: writable && !active && controlCapability(capabilities, 'startTraining'), resumeLatest: canResume, stop: writable && active && !finishing && controlCapability(capabilities, 'stopAndSave'), mode: writable && !active && modes.length > 0 && controlCapability(capabilities, 'switchMode'),
      budget: writable && !active, preview: writable && legacy.preview_available !== false && controlCapability(capabilities, 'preview'), previewPause: writable && legacy.suspended === false && controlCapability(capabilities, 'preview'),
      previewReset: writable && legacy.suspended === false && controlCapability(capabilities, 'preview'),
      trainingPause: writable && active && protocol && !finishing && !pauseRequested && !resuming && controlCapability(capabilities, 'pauseTraining'),
      saveSnapshot: writable && active && protocol && !finishing && !savePending && controlCapability(capabilities, 'saveSnapshot'),
      historyResume: writable && !active && controlCapability(capabilities, 'resumeSelected'), multiWindow: writable && controlCapability(capabilities, 'multipleViewports'),
      directory: writable && !active && storage?.locked !== true && localDirectories?.select === true && controlCapability(capabilities, 'directoryBridge'),
      directoryOpen: writable && localDirectories?.open === true && controlCapability(capabilities, 'directoryBridge'),
      modelImport: writable && localDirectories?.select === true && controlCapability(capabilities, 'directoryBridge'),
      ppoWrite: writable && !active && parameters?.locked !== true && controlCapability(capabilities, 'parameterWrites') },
    preview: { known, available: legacy?.preview_available !== false, open: legacy?.suspended === false, paused: legacy?.paused === true,
      syncStatus: legacy?.sync_status || null, syncError: legacy?.sync_error || legacy?.error || null,
      checkpoint: legacy?.checkpoint || null, runId: legacy?.model_run_id || null, steps: number(legacy?.preview_training_steps),
      syncedAt: producerTime(legacy?.preview_synced_at), modelSource: legacy?.model_source || null,
      episodeSeconds: number(legacy?.episode_seconds), frames: nonnegative(legacy?.frames), fps: number(legacy?.fps),
      inputs: Array.isArray(legacy?.actuator_inputs) ? legacy.actuator_inputs.map(number) : Array.isArray(legacy?.torques) ? legacy.torques.map(number) : [],
      units: Array.isArray(legacy?.input_units) ? legacy.input_units : Array.isArray(matchedDescriptor?.action?.units) ? matchedDescriptor.action.units : [],
      order: Array.isArray(matchedDescriptor?.action?.order) ? matchedDescriptor.action.order : [], description: legacy?.actuator_description || legacy?.action_semantics || null },
    controlHz: number(legacy?.control_hz ?? mode?.controlHz), obsDim: number(legacy?.obs_dim ?? mode?.obsDim),
    warnings: [...warnings, ...[dto, descriptor, checkpoints].flatMap(data => Array.isArray(data?.warnings) ? data.warnings.map(item => item.message || item.code) : []), ...(dto && !dtoMatches ? ['累计记录身份未与当前监督器匹配，显示本轮进度'] : []), ...(descriptor && !descriptorMatches ? ['运行描述身份未与当前监督器匹配，未用于当前配置'] : [])].filter(Boolean),
    candidates: Array.isArray(checkpoints?.items) ? checkpoints.items : [],
  };
}
export function commandRequest(action, view, { durationSeconds = 600, resume = false, mode = null, checkpoint = null, parameters = {}, requestId = null } = {}) {
  const requests = {
    start: ['start', '/training', { action: 'start', duration_seconds: durationSeconds, resume: Boolean(resume), checkpoint, parameters: {} }],
    stop: ['stop', '/training', { action: 'stop' }],
    preview: ['preview', '/control', { action: view.preview.open ? 'suspend' : 'resume' }],
    previewPause: ['previewPause', '/control', { action: view.preview.paused ? 'unpause' : 'pause' }],
    previewReset: ['previewReset', '/control', { action: 'reset' }],
    mode: ['mode', '/control', { action: 'switch-mode', mode }],
    trainingPause: ['trainingPause', '/training', { action: view.paused ? 'resume' : 'pause', ...(requestId ? { request_id: requestId } : {}) }],
    saveSnapshot: ['saveSnapshot', '/training', { action: 'save', ...(requestId ? { request_id: requestId } : {}) }],
  };
  const request = requests[action];
  if (!request || !view.controls[request[0]]) throw new Error('当前状态不支持此操作');
  if (action === 'start' && (resume && checkpoint === null && !view.controls.resumeLatest || !finite(durationSeconds) || durationSeconds < 0 || durationSeconds > 604800 || durationSeconds > 0 && durationSeconds < 6)) throw new Error('训练预算或续训起点不可用');
  if (action === 'start' && checkpoint !== null) {
    if (!view.controls.historyResume || !view.options.some(item => item.path === checkpoint && item.compatible === true)) throw new Error('所选模型尚未通过兼容性验证');
    request[2].checkpoint = checkpoint; request[2].resume = true;
  }
  if (action === 'start' && Object.keys(parameters).length) {
    if (!view.controls.ppoWrite) throw new Error('当前不能编辑训练参数');
    const checked = validateParameterDraft(view.parameterFields, parameters);
    if (!checked.valid) throw new Error(checked.errors.join('；'));
    request[2].parameters = checked.values;
  }
  if (action === 'mode' && !view.modes.some(item => item.id === mode)) throw new Error('未知动作接口');
  return { path: request[1], body: request[2] };
}
