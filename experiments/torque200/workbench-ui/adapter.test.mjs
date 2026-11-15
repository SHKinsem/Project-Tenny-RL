import test from 'node:test';
import assert from 'node:assert/strict';
import { buildWorkbenchView, checkpointLabel, commandRequest, loadingView, metricRecords, normalizeLearning, previewImageFailure, runId, validateParameterDraft } from './adapter.mjs';
import { createChartModel } from './chart-model.mjs';
import { apiJSON, createWorkbenchTransport } from './transport.mjs';

const catalog = { currentMode: 'leg_force_bc', modes: [{ id: 'leg_force_bc', label: '服务端腿力模式', controlHz: 100, obsDim: 24 }, { id: 'future_robot', label: '未来机器人', controlHz: 250, obsDim: 31 }] };
function state(overrides = {}) {
  return { mode: 'leg_force_bc', control_hz: 100, obs_dim: 24, suspended: true, paused: false, preview_available: true,
    training: { active: false, phase: 'stopped', run_id: 'run-2', run_dir: '/runs/run-2', elapsed_seconds: 60, duration_seconds: 600, steps: 1234, can_resume: true, latest_checkpoint: '/runs/run-1/checkpoint_8192', latest: { run_id: 'run-1', actual_timesteps: 8192, num_envs: 128 }, status: { num_envs: 512, runtime_profile: { requested: 'auto', effective: 'fast', effective_settings: { num_envs: 512, batch_size: 512, num_minibatches: 1, unroll_length: 16, num_updates_per_batch: 1 } } } }, ...overrides };
}
const connected = legacy => ({ legacy, catalog, connected: true, capabilities: { legacyControls: { startTraining: true, resumeLatest: true, stopAndSave: true, preview: true, switchMode: true } } });
function dto(overrides = {}) { return { available: true, mode: 'leg_force_bc', run: { id: 'run-2', reportedPhase: 'training', active: null, elapsedSeconds: 0, durationSeconds: 600 }, progress: { baseTrainedSeconds: 600, trainedSeconds: 600, targetTrainedSeconds: 1200, trainedSteps: 8192, cumulativePercent: 50, segmentPercent: 0 }, ...overrides }; }

test('启动与断线不授予写入能力，也不制造默认模式或计数', () => {
  const boot = buildWorkbenchView();
  assert.equal(boot.mode, null); assert.equal(boot.config.num_envs, null); assert.equal(boot.progress.trainedSeconds, null);
  assert.ok(Object.values(boot.controls).every(value => value === false));
  const offline = buildWorkbenchView({ ...connected(state()), connected: false, error: 'offline' });
  assert.equal(offline.progress.elapsedSeconds, 60); assert.equal(offline.phaseLabel, '连接中断 · 保留最后报告');
  assert.ok(Object.values(offline.controls).every(value => value === false));
});
test('模式目录是唯一标签来源，支持新机器人而无需前端添加模式分支', () => {
  const view = buildWorkbenchView(connected(state({ mode: 'future_robot', control_hz: undefined, obs_dim: undefined })));
  assert.equal(view.mode, 'future_robot'); assert.equal(view.modeLabel, '未来机器人'); assert.equal(view.controlHz, 250); assert.equal(view.obsDim, 31);
  assert.equal(commandRequest('mode', view, { mode: 'leg_force_bc' }).body.mode, 'leg_force_bc');
  const missing = buildWorkbenchView({ legacy: state(), connected: true });
  assert.equal(missing.modeLabel, 'leg_force_bc'); assert.equal(missing.controls.mode, false);
});
test('活动与可恢复能力只来自监督器，磁盘 reportedPhase 和候选不能授权', () => {
  const legacy = state(); legacy.training.can_resume = false;
  const view = buildWorkbenchView({ ...connected(legacy), dto: dto(), checkpoints: { items: [{ canResume: true, verification: 'verified', relativePath: 'other' }] } });
  assert.equal(view.active, false); assert.equal(view.phase, 'stopped'); assert.equal(view.controls.stop, false); assert.equal(view.controls.resumeLatest, false);
  assert.throws(() => commandRequest('start', view, { resume: true }), /不可用/);
});
test('当前运行身份不被 latest 的旧身份替代，run_id 优先于目录', () => {
  assert.equal(runId({ run_id: 'failed-run', run_dir: 'C:\\runs\\different', latest: { run_id: 'old-run' } }), 'failed-run');
  assert.equal(runId({ run_dir: 'C:\\runs\\prepared-run\\' }), 'prepared-run');
  const wrong = buildWorkbenchView({ ...connected(state()), dto: dto({ run: { id: 'run-1', elapsedSeconds: 500, durationSeconds: 600 } }) });
  assert.equal(wrong.dtoMatches, false); assert.equal(wrong.progress.trainedSeconds, null); assert.equal(wrong.progress.elapsedSeconds, 60);
});
test('实际 512 环境及继承配置优先，旧模型 128 不冒充当前运行', () => {
  const view = buildWorkbenchView(connected(state()));
  assert.deepEqual([view.config.num_envs, view.config.batch_size, view.config.num_minibatches, view.config.unroll_length, view.config.num_updates_per_batch], [512, 512, 1, 16, 1]);
  const legacy = state(); legacy.training.status = {}; legacy.training.training_profile = 'standing_v2';
  const unknown = buildWorkbenchView(connected(legacy));
  assert.equal(unknown.config.num_envs, null); assert.equal(unknown.config.runtime, null); assert.equal(unknown.config.learning_rate, null);
});
test('保存描述必须匹配 mode 和当前 run 才能补充只读配置', () => {
  const descriptor = { available: true, mode: 'leg_force_bc', runId: 'run-2', robot: { id: null }, action: { order: ['leg_a', 'wheel_b'], units: ['rad', 'N·m'] }, runtime: { learningRate: 0.0002, numEnvs: 16 }, observation: { actorDim: 24, criticDim: 31 } };
  const view = buildWorkbenchView({ ...connected(state()), descriptor });
  assert.equal(view.config.learning_rate, 0.0002); assert.equal(view.config.num_envs, 512); assert.equal(view.descriptor.robot.id, null);
  assert.deepEqual(view.preview.order, ['leg_a', 'wheel_b']);
  const wrong = buildWorkbenchView({ ...connected(state()), descriptor: { ...descriptor, runId: 'run-1' } });
  assert.equal(wrong.descriptor, null); assert.equal(wrong.config.learning_rate, null);
});
test('累计 600 秒追加 600 秒从 50% 开始，且使用已匹配的正式计时', () => {
  const initial = buildWorkbenchView({ ...connected(state()), dto: dto() });
  assert.equal(initial.progress.percent, 50); assert.equal(initial.progress.elapsedSeconds, 0); assert.equal(initial.progress.clock, '10:00 / 20:00');
  const nextDto = dto(); nextDto.run.elapsedSeconds = 60; nextDto.progress.trainedSeconds = 660; nextDto.progress.cumulativePercent = 55; nextDto.progress.segmentPercent = 10;
  const next = buildWorkbenchView({ ...connected(state()), dto: nextDto });
  assert.equal(next.progress.percent, 55); assert.equal(next.progress.segmentPercent, 10);
});
test('未知历史保持累计 null，本轮预算可用；准备期不能计入正式训练', () => {
  const report = dto(); report.run.elapsedSeconds = 60; report.progress = { baseTrainedSeconds: null, trainedSeconds: null, trainedSteps: null, targetTrainedSeconds: null, cumulativePercent: null, segmentPercent: 10 };
  const view = buildWorkbenchView({ ...connected(state()), dto: report });
  assert.equal(view.progress.trainedSeconds, null); assert.equal(view.progress.percent, 10); assert.equal(view.progress.label, '本轮 / 预算时长');
  const legacy = state(); legacy.training.active = true; legacy.training.training_clock_started = false;
  const preparing = buildWorkbenchView(connected(legacy));
  assert.equal(preparing.progress.elapsedSeconds, 0); assert.equal(preparing.phase, 'preparing');
});
test('学习窗口保留原始缺失、负值、微小量；不编造评估时间或跨 run 拼接', () => {
  const data = { available: true, metadata: { run: 'console/leg_force_bc/run-2' }, points: [
    { steps: 1, episode_seconds: 0.3, episode_reward: 2, value_loss: 1e-10, policy_loss: -1e-9 },
    { steps: 2, episode_seconds: null, value_loss: null, policy_loss: 0 },
    { steps: 3, episode_seconds: 0.4, value_loss: 2e-10, policy_loss: -2e-9 },
  ] };
  const result = normalizeLearning(data, state(), catalog);
  assert.equal(result.points[0].at, null); assert.equal(result.points[0].valueLoss, 1e-10); assert.equal(result.points[0].policyLoss, -1e-9);
  assert.equal(result.points[1].episodeSeconds, null); assert.equal(createChartModel(result.points, 'episodeSeconds').segments.length, 2);
  assert.equal(normalizeLearning(data, state({ training: { run_id: 'run-3' } }), catalog).points.length, 0);
  assert.equal(normalizeLearning({ ...data, metadata: { run: 'future_robot/run-2' } }, state(), catalog).points.length, 0);
});

test('独立指标按 measured 采样，loss-only 与末尾空事件不切断评估或抹掉最新测量', () => {
  const data = { available: true, metadata: { run: 'leg_force_bc/run-2', eval_point_count: 3 }, points: [
    { steps: 0, episode_seconds: 0.213, episode_reward: 1.234, measured: ['episode_seconds', 'episode_reward'] },
    { steps: 1, value_loss: 1e-10, policy_loss: -1e-9, measured: ['value_loss', 'policy_loss'] },
    { steps: 2, episode_seconds: 0.254, episode_reward: 2.345, measured: ['episode_seconds', 'episode_reward'] },
    { steps: 3, value_loss: 2e-10, measured: ['value_loss'] },
    { steps: 4, episode_seconds: 0.387, episode_reward: 3.456, measured: ['episode_seconds', 'episode_reward'] },
    { steps: 5, value_loss: 3e-10, measured: ['value_loss'] },
    { steps: 6, measured: [] },
  ] };
  const { points, metadata } = normalizeLearning(data, state(), catalog);
  const duration = metricRecords(points, 'episodeSeconds'), loss = metricRecords(points, 'valueLoss');
  assert.deepEqual(duration.map(point => point.steps), [0, 2, 4]); assert.equal(createChartModel(duration, 'episodeSeconds').segments[0].length, 3);
  assert.equal(duration.at(-1).episodeSeconds, 0.387); assert.equal(loss.at(-1).valueLoss, 3e-10); assert.equal(loss.at(-1).steps, 5);
  assert.deepEqual(points[0].measured, ['episodeSeconds', 'reward']); assert.equal(metadata.eval_point_count, 3); assert.equal(duration.at(-1).at, null);
});

test('显式无效测量与 gap 保留断线，旧无 measured 数据继续保守保留空记录', () => {
  const data = { available: true, metadata: { run: 'run-2' }, points: [
    { steps: 0, episode_seconds: 0.2, measured: ['episode_seconds'] },
    { steps: 1, value_loss: 0.01, measured: ['value_loss'] },
    { steps: 2, episode_seconds: NaN, measured: ['episode_seconds'] },
    { steps: 3, episode_seconds: 0.4, measured: ['episode_seconds'] },
    { steps: 4, episode_seconds: null, measured: ['episode_seconds'] },
    { steps: 5, measured: [] },
  ] };
  const { points } = normalizeLearning(data, state(), catalog), duration = metricRecords(points, 'episodeSeconds');
  assert.deepEqual(duration.map(point => point.steps), [0, 2, 3, 4]); assert.equal(duration.at(-1).episodeSeconds, null);
  assert.equal(createChartModel(duration, 'episodeSeconds').segments.length, 2);
  const legacy = points.map(({ measured, ...point }) => point);
  assert.equal(metricRecords(legacy, 'episodeSeconds').length, points.length); assert.equal(metricRecords(legacy, 'episodeSeconds').at(-1).steps, 5);
  const gaps = normalizeLearning({ ...data, points: [{ steps: 0, measured: [], gap: ['value_loss'] }, { steps: 1, measured: [], gap: true }] }, state(), catalog).points;
  assert.equal(metricRecords(gaps, 'valueLoss').length, 2); assert.equal(metricRecords(gaps, 'episodeSeconds').length, 1);
});
test('命令精确区分训练停止保存、预览暂停和未实现的训练暂停/快照', () => {
  const legacy = state(); legacy.training.active = true; legacy.training.phase = 'training'; legacy.suspended = false;
  const view = buildWorkbenchView(connected(legacy));
  assert.deepEqual(commandRequest('stop', view), { path: '/training', body: { action: 'stop' } });
  assert.deepEqual(commandRequest('previewPause', view), { path: '/control', body: { action: 'pause' } });
  assert.deepEqual(commandRequest('preview', view), { path: '/control', body: { action: 'suspend' } });
  for (const unsupported of ['trainingPause', 'saveSnapshot', 'historyResume']) assert.throws(() => commandRequest(unsupported, view), /不支持/);
  const idle = buildWorkbenchView(connected(state()));
  assert.deepEqual(commandRequest('start', idle, { resume: true, durationSeconds: 120 }), { path: '/training', body: { action: 'start', duration_seconds: 120, resume: true, checkpoint: null, parameters: {} } });
});
test('checkpoint 名称来自真实 alias、步数或文件尾部，不依赖候选数组排序', () => {
  assert.equal(checkpointLabel('/runs/old/checkpoint_8192'), '8,192 步');
  assert.equal(checkpointLabel('/runs/policy.bin', 16384), '16,384 步');
  assert.equal(checkpointLabel('/runs/policy.bin', 16384, 'balance-a'), 'balance-a');
  assert.equal(checkpointLabel(null), '未报告');
  const source = state(); const before = JSON.stringify(source); buildWorkbenchView({ ...connected(source), dto: dto() }); assert.equal(JSON.stringify(source), before);
});

test('持久 checkpoint UUID 优先显示短身份，旧模型明确标注本轮捕获步数而非累计', () => {
  const legacy = state(), uuid = 'ab12cd34-5678-4abc-9def-123456789abc', candidateId = 'cd34ef56-7890-4abc-8def-123456789abc';
  legacy.training.latest_checkpoint = '/runs/run-1/checkpoint_10000';
  Object.assign(legacy.training.latest, { actual_timesteps: 10000, workbench_checkpoint_id: uuid, alias: 'saved-name' });
  const report = dto(); report.progress.trainedSteps = 110000;
  const checkpoints = { items: [{ relativePath: 'run-1/checkpoint_10000', id: candidateId, capturedSteps: 10000, trainedSteps: 110000 }] };
  const view = buildWorkbenchView({ ...connected(legacy), dto: report, checkpoints });
  assert.equal(view.checkpointLabel, 'CP·AB12CD34'); assert.equal(view.checkpointId, uuid); assert.equal(view.progress.trainedSteps, 110000);
  assert.ok(view.checkpointTitle.includes(uuid)); assert.ok(view.checkpointTitle.includes(legacy.training.latest_checkpoint));
  assert.equal(legacy.training.latest.alias, 'saved-name');
  legacy.training.latest.workbench_checkpoint_id = uuid.toUpperCase();
  assert.equal(buildWorkbenchView({ ...connected(legacy), checkpoints }).checkpointLabel, 'CP·CD34EF56');
  delete legacy.training.latest.alias;
  legacy.training.latest.workbench_checkpoint_id = '00000000-0000-0000-0000-000000000000';
  const old = buildWorkbenchView({ ...connected(legacy), dto: report });
  assert.equal(old.checkpointLabel, '10,000 步'); assert.ok(old.checkpointTitle.includes('本轮捕获步数：10,000'));
  assert.ok(!old.checkpointTitle.includes('110,000'));
});

test('轮询请求串行，命令等待提交后的重新读取；失败保持最后计数', async () => {
  let release, entered, stateReads = 0, concurrent = 0, maximum = 0, fail = false, received;
  const firstRead = new Promise(resolve => { entered = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const fetchImpl = async (path, options = {}) => {
    if (path === '/state') {
      stateReads++; concurrent++; maximum = Math.max(maximum, concurrent);
      if (stateReads === 1) { entered(); await gate; }
      concurrent--;
      if (fail) throw new Error('offline');
      return { ok: true, json: async () => state() };
    }
    return { ok: true, json: async () => options.method === 'POST' ? { ok: true } : {} };
  };
  const transport = createWorkbenchTransport({ fetchImpl, intervalMs: 60000, onSnapshot: value => { received = value; } });
  try {
    transport.start(); await firstRead;
    const afterCommand = transport.command({ path: '/control', body: { action: 'reset' } });
    await Promise.resolve(); release(); await afterCommand;
    assert.equal(stateReads, 2); assert.equal(maximum, 1); assert.equal(received.connected, true);
    fail = true; await transport.refresh();
    assert.equal(received.connected, false); assert.equal(received.legacy.training.steps, 1234);
  } finally { transport.stop(); }
});

test('保存收尾阶段禁用重复停止，v1 与旧接口错误均保留可读消息', async () => {
  for (const [phase, label] of [['exporting', '导出中'], ['verifying', '核验中']]) {
    const legacy = state(); legacy.training.active = true; legacy.training.phase = phase;
    const view = buildWorkbenchView(connected(legacy));
    assert.equal(view.controls.stop, false); assert.equal(view.phaseLabel, label);
  }
  for (const error of ['旧接口失败', { code: 'not_available', message: '记录暂不可用' }]) {
    await assert.rejects(apiJSON('/unused', { fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({ error }) }) }), { message: typeof error === 'string' ? error : error.message });
  }
});

const completionCapabilities = { legacyControls: { startTraining: true, resumeLatest: true, stopAndSave: true, preview: true, switchMode: true, pauseTraining: true, saveSnapshot: true, resumeSelected: true, parameterWrites: true, multipleViewports: true, directoryBridge: true } };
test('加载阶段无分母时不生成百分比，真实分母与错误保留各自语义', () => {
  const unknown = loadingView({ active: true, stage: 'loading_environment', progress: null });
  assert.equal(unknown.label, '加载环境'); assert.equal(unknown.percent, null); assert.equal(unknown.indeterminate, true);
  assert.equal(loadingView({ active: true, completed: 2, total: 5 }).percent, 40);
  assert.equal(loadingView({ active: true, completed: 0, total: 0 }).percent, null);
  const error = loadingView({ active: true, stage: 'error', error: '模型读取失败' });
  assert.equal(error.active, false); assert.equal(error.visible, true); assert.equal(error.label, '模型读取失败');
  assert.equal(buildWorkbenchView().activity.stage, 'connecting'); assert.equal(buildWorkbenchView().activity.percent, null);
});
test('首帧请求失败停止不定进度并保留错误，旧请求与关闭代次不能覆盖新窗口', () => {
  const request = { key: '/stream.mjpg?viewport=1&generation=3', generation: 3 };
  const current = { ...request, open: true, connected: true, intendedOpen: true, loading: { active: true, stage: 'first_frame' } };
  const failure = previewImageFailure(request, current);
  assert.equal(failure.active, false); assert.equal(failure.indeterminate, false); assert.equal(failure.visible, true); assert.equal(failure.stage, 'error'); assert.match(failure.label, /失败.*重试/);
  assert.equal(previewImageFailure(request, { ...current, intendedOpen: false }), null);
  assert.equal(previewImageFailure(request, { ...current, key: null, open: false }), null);
  assert.equal(previewImageFailure(request, { ...current, key: '/stream.mjpg?viewport=1&generation=5', generation: 5 }), null);
  assert.equal(previewImageFailure(request, { ...current, generation: 5 }), null);
  const retry = { key: '/stream.mjpg?viewport=1&generation=5', generation: 5 };
  assert.notEqual(previewImageFailure(retry, { ...current, ...retry }), null);
});
test('旧训练不获得新协议控制，新协议暂停在确认前后准确区分并能继续', () => {
  const legacy = state(); legacy.training.active = true; legacy.training.phase = 'training';
  const source = { ...connected(legacy), capabilities: completionCapabilities };
  assert.equal(buildWorkbenchView(source).controls.trainingPause, false); assert.equal(buildWorkbenchView(source).controls.saveSnapshot, false);
  legacy.training.workbench_control_protocol = 1;
  let view = buildWorkbenchView(source);
  assert.equal(view.controls.trainingPause, true); assert.deepEqual(commandRequest('trainingPause', view).body, { action: 'pause' });
  legacy.training.pause_requested = true; legacy.training.workbench_control_result = { requestId: 'request-1', action: 'pause', status: 'pending' };
  view = buildWorkbenchView(source); assert.equal(view.phase, 'pausing'); assert.equal(view.controls.trainingPause, false);
  legacy.training.pause_requested = false; legacy.training.paused = true; legacy.training.workbench_control_result.status = 'applied';
  view = buildWorkbenchView(source); assert.equal(view.phase, 'paused'); assert.equal(view.controls.saveSnapshot, true); assert.equal(view.controls.stop, true);
  assert.deepEqual(commandRequest('trainingPause', view).body, { action: 'resume' });
});
test('保存只按异步回执完成，pending 不冒充 completed 且不停止训练', () => {
  const legacy = state(); Object.assign(legacy.training, { active: true, phase: 'training', workbench_control_protocol: 1 });
  const source = { ...connected(legacy), capabilities: completionCapabilities };
  let view = buildWorkbenchView(source); assert.deepEqual(commandRequest('saveSnapshot', view, { requestId: 'id-1' }).body, { action: 'save', request_id: 'id-1' });
  legacy.training.workbench_save_results = [{ requestId: 'id-1', status: 'pending' }];
  view = buildWorkbenchView(source); assert.equal(view.savePending, true); assert.equal(view.controls.saveSnapshot, false); assert.equal(view.active, true); assert.equal(view.save.checkpoint, undefined);
  legacy.training.workbench_save_results[0] = { requestId: 'id-1', status: 'completed', checkpoint: '/runs/snapshot/checkpoint_1234', capturedSteps: 1234 };
  view = buildWorkbenchView(source); assert.equal(view.savePending, false); assert.equal(view.save.status, 'completed'); assert.equal(view.controls.saveSnapshot, true); assert.equal(view.active, true);
  legacy.training.workbench_save_results.push({ requestId: 'id-2', status: 'failed', error: 'export failed' });
  view = buildWorkbenchView(source); assert.equal(view.save.status, 'failed'); assert.equal(view.save.error, 'export failed');
});
test('历史模型使用兼容选项的精确路径，即使最新模型不可续训也不偷偷替换', () => {
  const legacy = state(); legacy.training.can_resume = false;
  const path = '/verified/history/checkpoint_42';
  const source = { ...connected(legacy), capabilities: completionCapabilities, checkpointOptions: { mode: 'leg_force_bc', checkpoints: [{ path, compatible: true }, { path: '/invalid', compatible: false, reason: 'observation mismatch' }] } };
  const view = buildWorkbenchView(source), request = commandRequest('start', view, { resume: true, checkpoint: path });
  assert.equal(request.body.checkpoint, path); assert.equal(request.body.resume, true); assert.notEqual(request.body.checkpoint, view.checkpoint);
  assert.throws(() => commandRequest('start', view, { checkpoint: '/invalid' }), /兼容性/);
  assert.throws(() => commandRequest('start', view, { checkpoint: '/unlisted' }), /兼容性/);
});
test('参数只提交编辑项并校验类型/边界/只读字段，活动训练保持锁定', () => {
  const fields = [
    { key: 'num_envs', label: '并行环境', type: 'integer', min: 1, max: 1024, editable: true },
    { key: 'learning_rate', label: '学习率', type: 'number', min: 1e-8, max: 1, editable: true },
    { key: 'enabled', label: '开关', type: 'boolean', editable: true },
    { key: 'gae_lambda', label: 'GAE', type: 'number', editable: false },
  ];
  assert.deepEqual(validateParameterDraft(fields, { num_envs: '256', enabled: false }).values, { num_envs: 256, enabled: false });
  for (const draft of [{ num_envs: '1.2' }, { num_envs: '' }, { num_envs: 2048 }, { learning_rate: NaN }, { enabled: 'false' }, { gae_lambda: 0.9 }, { unknown: 1 }]) assert.equal(validateParameterDraft(fields, draft).valid, false);
  const legacy = state(), source = { ...connected(legacy), capabilities: completionCapabilities, parameters: { mode: legacy.mode, fields, locked: false } };
  const request = commandRequest('start', buildWorkbenchView(source), { parameters: { num_envs: '256' } }); assert.deepEqual(request.body.parameters, { num_envs: 256 });
  legacy.training.active = true; assert.equal(buildWorkbenchView(source).controls.ppoWrite, false);
});
test('目录与多窗能力依赖真实桥接，活动训练只锁保存位置而非打开与导入', () => {
  const legacy = state(), source = { ...connected(legacy), capabilities: completionCapabilities, storage: { locked: false }, localDirectories: { select: true, open: true } };
  let view = buildWorkbenchView(source); assert.equal(view.controls.directory, true); assert.equal(view.controls.multiWindow, true);
  legacy.training.active = true; view = buildWorkbenchView(source); assert.equal(view.controls.directory, false); assert.equal(view.controls.directoryOpen, true); assert.equal(view.controls.modelImport, true);
  view = buildWorkbenchView({ ...source, localDirectories: { available: false, select: false, open: false } }); assert.equal(view.controls.directoryOpen, false); assert.equal(view.controls.modelImport, false);
});
test('切换参数起点时丢弃迟到的旧 schema，不覆盖新起点草稿', async () => {
  let release, entered, received, reads = 0, resolveNewSchema;
  const sawNewSchema = new Promise(resolve => { resolveNewSchema = resolve; });
  const first = new Promise(resolve => { entered = resolve; }), gate = new Promise(resolve => { release = resolve; });
  const fetchImpl = async path => {
    let data = {};
    if (path === '/state') data = state();
    else if (path.startsWith('/api/workbench/v1/parameters')) {
      reads++; if (reads === 1) { entered(); await gate; }
      const checkpoint = new URL(path, 'http://fixture.invalid').searchParams.get('checkpoint'); data = { mode: 'leg_force_bc', checkpoint, fields: [] };
    }
    return { ok: true, json: async () => data };
  };
  const transport = createWorkbenchTransport({ fetchImpl, intervalMs: 60000, onSnapshot: value => { received = value; if (value.parameters?.checkpoint === '/verified/B') resolveNewSchema(value); } });
  try { transport.start(); await first; transport.setParameterCheckpoint('/verified/B'); const refreshed = transport.refresh(); release(); await refreshed; await sawNewSchema; assert.equal(received.parameters.checkpoint, '/verified/B'); }
  finally { transport.stop(); }
});
