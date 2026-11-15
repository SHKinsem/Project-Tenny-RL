// Standalone UI prototype: synthetic data, pure in-memory transitions, no I/O.
export const SCENARIOS = [
  ['training', '训练进行中'], ['paused', '训练已暂停'], ['snapshot-saving', '快照保存中'],
  ['resume-progress', '继续训练 · 10分钟＋10分钟'],
  ['ready', '空闲 · 可继续训练'], ['empty', '首次使用 · 无模型'],
  ['preparing', '准备检查 · 部分复用'], ['prepare-failed', '准备检查失败'],
  ['prepare-cancelled', '准备已取消'], ['compiling', '编译中 · 时长未知'],
  ['unlimited', '不限时训练'], ['stopping', '正在停止'], ['saving', '保存中'],
  ['exporting', '导出中'], ['verifying', '校验中'], ['completed', '完成 · 已验证'],
  ['failed', '训练中断'], ['save-failed', '保存校验失败'], ['disconnected', '连接断开'],
  ['stale', '已连接 · 数据延迟'], ['preview-syncing', '预览同步中'],
  ['preview-failed', '预览同步失败'], ['preview-running', '预览已开启'],
  ['preview-paused', '预览已暂停'], ['preview-ended', '预览回合结束'],
].map(([id, label]) => ({ id, label }));

const START = Date.parse('2026-11-15T12:45:00+08:00');
export const ACTIVE_PHASES = ['preparing', 'compiling', 'training', 'paused', 'stopping', 'saving', 'exporting', 'verifying'];
const ACTIVE = new Set(ACTIVE_PHASES);
const SAVING = ['stopping', 'saving', 'exporting', 'verifying'];
const clone = value => structuredClone(value);
const round = (value, digits = 3) => Number(value.toFixed(digits));
const checkpointBusy = state => ['saving', 'verifying'].includes(state.checkpointSave.phase);
const knownAmount = value => Number.isFinite(value) && value >= 0 ? value : null;
const cumulative = (base, added) => knownAmount(base) === null ? null : base + added;

function event(state, title, detail = '') {
  state.events.unshift({ id: `demo-event-${state._clock.event++}`, at: state.now, title, detail });
  state.events = state.events.slice(0, 14);
}

function model(id, name, steps, offset, extra = {}) {
  return { id, name, mode: 'original', controlHz: 100, steps, savedAt: START - offset,
    verified: true, compatible: true, reason: '', sourceLabel: '已验证模型',
    trainedSeconds: null, trainedSteps: null, ...extra };
}

function base() {
  const points = [0.22, 0.28, 0.31, 0.38, 0.36, 0.43, 0.48, 0.54].map((value, i) => ({
    steps: i * 81920, episodeSeconds: value, reward: [12.4, 18.9, 22.5, 34.8, 29.1, 42.7, 48.2, 56.3][i],
    valueLoss: i === 0 ? null : round(0.71 - i * 0.043),
    policyLoss: i === 0 ? null : round(-0.004 + Math.sin(i) * 0.002, 5),
    at: START - (7 - i) * 45000 - 18000,
  }));
  return {
    demo: true, scenario: 'training', now: START, connection: 'connected', phase: 'training',
    run: { id: 'demo-run-024', name: '基线平衡 · 024', mode: 'original',
      modeLabel: '原版 · 100 Hz', actionLabel: '腿位置＋轮速度', controlHz: 100, obsDim: 26,
      steps: 581632, schedule: 'timed', durationSeconds: 900, elapsedSeconds: 248,
      parentModelId: 'demo-model-023', baseTrainedSeconds: 600, baseTrainedSteps: 524288 },
    preparation: { elapsedSeconds: 14, stages: [
      { id: 'validation', label: '动态预检', status: 'cached', seconds: 0, detail: '复用已通过记录 · R01' },
      { id: 'first', label: '新训练与导出验证', status: 'cached', seconds: 0, detail: '复用已通过记录 · R02' },
      { id: 'restart', label: '热启动验证', status: 'cached', seconds: 0, detail: '绑定上游模型 · R03' },
    ] },
    data: { receivedAt: START - 1000, progressAt: START - 1000, evaluationAt: START - 18000,
      evaluationSteps: points.at(-1).steps, evalCount: points.length, evalEnvs: 8 },
    evaluation: { targetSeconds: 6, points },
    models: [
      model('demo-model-023', '基线平衡 · 023', 524288, 3600000, { trainedSeconds: 600, trainedSteps: 524288 }),
      model('demo-model-019', '基线平衡 · 019', 327680, 86400000, { trainedSeconds: 360, trainedSteps: 327680 }),
      model('demo-author-policy', '作者预训练策略', 0, 172800000, { sourceLabel: '作者预训练策略' }),
      model('demo-torque-policy', '全扭矩实验 · 200 Hz', 262144, 7200000,
        { mode: 'torque200', controlHz: 200, compatible: false, reason: '动作定义、频率与观测维度不同' }),
    ],
    selectedModelId: 'demo-model-023', latestSavedModelId: 'demo-model-023',
    checkpointSave: { phase: 'idle', steps: null, requestedAt: null, modelId: null, error: null },
    preview: { status: 'off', modelId: null, runId: null, steps: null, syncedAt: null, sampleAt: null,
      telemetry: { episodeSeconds: null, resets: null, leftLeg: null, rightLeg: null, leftWheel: null, rightWheel: null }, error: null },
    events: [
      { id: 'demo-event-2', at: START - 248000, title: '训练已开始', detail: '从基线平衡 · 023 继续；优化器重新初始化' },
      { id: 'demo-event-1', at: START - 262000, title: '运行环境准备完成', detail: '原版运行环境已就绪' },
    ],
    error: null, notice: null,
    _clock: { phase: 0, preview: 0, evaluation: 18, event: 3, run: 25, checkpointSave: 0,
      checkpointElapsedSeconds: 0, snapshot: 1 },
  };
}

function clearRun(state) {
  state.run.steps = 0;
  state.run.elapsedSeconds = 0;
  state.run.parentModelId = null;
  state.run.baseTrainedSeconds = 0;
  state.run.baseTrainedSteps = 0;
  state.run.id = 'demo-next-run';
  state.run.name = '下一轮训练';
  state.evaluation.points = [];
  state.data.evaluationAt = null;
  state.data.evaluationSteps = null;
  state.data.evalCount = 0;
  state.events = [];
}

function preparationScenario(state, failed = false, cancelled = false) {
  clearRun(state);
  state.phase = failed ? 'failed' : cancelled ? 'cancelled' : 'preparing';
  state.run.mode = 'hybrid_virtual_leg_force_100';
  state.run.modeLabel = '混合动作 · 100 Hz';
  state.run.actionLabel = '腿力＋轮扭矩';
  state.run.obsDim = 24;
  state.preparation.elapsedSeconds = failed ? 17 : 9;
  state.preparation.stages[1] = { id: 'first', label: '新训练与导出验证',
    status: failed ? 'failed' : cancelled ? 'passed' : 'running', seconds: 9,
    detail: failed ? '导出校验未通过 · 已完成阶段仍保留' : cancelled ? '已通过 · R02' : '配置变化，重新核对导出结果' };
  state.preparation.stages[2] = { id: 'restart', label: '热启动验证',
    status: cancelled ? 'cancelled' : 'waiting', seconds: 0,
    detail: cancelled ? '尚未完成，下次只补缺失阶段' : '等待上游模型验证' };
  state.models = state.models.map(value => ({ ...value, compatible: false, reason: '与当前混合动作模式不同' }));
  state.selectedModelId = null;
  state.latestSavedModelId = null;
  state.error = failed ? '新训练与导出验证未通过。动态预检记录已保留，可重新准备。' : null;
  event(state, failed ? '准备检查失败' : cancelled ? '准备已取消' : '准备检查进行中', '逐阶段保留已完成记录');
}

function snapshot(state, status) {
  const saved = state.models.find(value => value.id === state.latestSavedModelId);
  const live = ['training', 'paused', 'stopping', 'saving', 'exporting', 'verifying'].includes(state.phase);
  const steps = live ? state.run.steps : saved?.steps ?? state.run.steps;
  state.preview = { ...state.preview, status, modelId: live ? `demo-snapshot-${steps}` : saved?.id ?? null,
    runId: live ? state.run.id : saved?.id ?? null, steps, syncedAt: state.now, sampleAt: state.now,
    telemetry: { episodeSeconds: 0.31, resets: 3, leftLeg: 0.0431, rightLeg: 0.0454, leftWheel: 0.0082, rightWheel: 0.0089 },
    error: null };
}

export function createDemo(scenario = 'training') {
  const state = base();
  state.scenario = SCENARIOS.some(value => value.id === scenario) ? scenario : 'training';
  switch (state.scenario) {
    case 'paused': state.phase = 'paused'; event(state, '训练已暂停', '计时、采样与评估暂停；预览状态独立'); break;
    case 'snapshot-saving': requestCheckpoint(state); break;
    case 'resume-progress':
      state.phase = 'paused';
      state.run.name = '继续训练 · 024';
      state.run.elapsedSeconds = 0; state.run.steps = 0; state.run.durationSeconds = 600;
      state.evaluation.points = []; state.data.evaluationAt = null;
      state.data.evaluationSteps = null; state.data.evalCount = 0;
      state._clock.evaluation = 0; state.events = [];
      event(state, '已加载父模型的累计训练量', '已训练 600 秒，本轮新增预算 600 秒；等待继续训练');
      break;
    case 'ready': state.phase = 'idle'; clearRun(state); break;
    case 'empty': state.phase = 'idle'; clearRun(state); state.models = []; state.selectedModelId = null; state.latestSavedModelId = null; break;
    case 'preparing': preparationScenario(state); break;
    case 'prepare-failed': preparationScenario(state, true); break;
    case 'prepare-cancelled': preparationScenario(state, false, true); break;
    case 'compiling': state.phase = 'compiling'; clearRun(state); event(state, '编译计算图', '此阶段完成时间未知，训练计时包含编译与评估'); break;
    case 'unlimited': state.run.schedule = 'unlimited'; state.run.durationSeconds = 0; break;
    case 'stopping': case 'saving': case 'exporting': case 'verifying': state.phase = state.scenario; event(state, '正在完成保存流程', '仅校验通过后发布新模型'); break;
    case 'completed': state.phase = 'completed'; state.run.elapsedSeconds = 900; complete(state); break;
    case 'failed': state.phase = 'failed'; state.error = '训练进程意外中断。最近已验证模型仍可继续使用。'; event(state, '训练中断', '已有模型和评估记录保留'); break;
    case 'save-failed': state.phase = 'failed'; state.error = '模型校验未通过；本轮模型尚未标记为已验证。最近已验证模型保留。'; event(state, '保存校验失败', '未发布新的已验证模型'); break;
    case 'disconnected':
      state.connection = 'disconnected'; state.data.receivedAt = START - 67000; state.data.progressAt = START - 67000;
      state.evaluation.points.forEach(point => { point.at -= 82000; });
      state.data.evaluationAt = state.evaluation.points.at(-1).at;
      break;
    case 'stale':
      state.connection = 'stale'; state.data.progressAt = START - 130000;
      state.evaluation.points.forEach(point => { point.at -= 152000; });
      state.data.evaluationAt = state.evaluation.points.at(-1).at;
      break;
    case 'preview-syncing': state.preview.status = 'syncing'; break;
    case 'preview-failed': state.preview.status = 'failed'; state.preview.error = '策略同步超时；训练状态不受影响。'; break;
    case 'preview-running': snapshot(state, 'running'); break;
    case 'preview-paused': snapshot(state, 'paused'); state.preview.syncedAt -= 30000; state.preview.sampleAt -= 22000; break;
    case 'preview-ended': snapshot(state, 'ended'); state.preview.telemetry.episodeSeconds = 0.54; break;
  }
  return state;
}

function complete(state) {
  state.phase = 'completed';
  const result = model(`demo-saved-${state.run.id}`, `${state.run.name} · 已保存`, state.run.steps, 0,
    { savedAt: state.now, mode: state.run.mode, controlHz: state.run.controlHz, sourceLabel: '本轮校验通过',
      trainedSeconds: cumulative(state.run.baseTrainedSeconds, state.run.elapsedSeconds),
      trainedSteps: cumulative(state.run.baseTrainedSteps, state.run.steps) });
  state.models.unshift(result);
  state.latestSavedModelId = result.id;
  event(state, '停止完成 · 模型已验证', '参数重载与导出一致性校验通过');
}

function requestCheckpoint(state) {
  state.checkpointSave = { phase: 'saving', steps: state.run.steps, requestedAt: state.now, modelId: null, error: null };
  state._clock.checkpointSave = 0;
  state._clock.checkpointElapsedSeconds = state.run.elapsedSeconds;
  event(state, '正在保存快照', '固定请求时的参数步数，训练状态保持不变');
}

function advanceCheckpoint(state) {
  if (!checkpointBusy(state)) return;
  state._clock.checkpointSave += 1;
  if (state._clock.checkpointSave < 2) return;
  state._clock.checkpointSave = 0;
  if (state.checkpointSave.phase === 'saving') {
    state.checkpointSave.phase = 'verifying';
    event(state, '正在校验快照', '校验完成前，不发布为已验证模型');
    return;
  }
  const id = `demo-checkpoint-${state.run.id}-${state._clock.snapshot++}`;
  const result = model(id, `${state.run.name} · 快照`, state.checkpointSave.steps, 0,
    { savedAt: state.now, mode: state.run.mode, controlHz: state.run.controlHz, sourceLabel: '快照校验通过',
      trainedSeconds: cumulative(state.run.baseTrainedSeconds, state._clock.checkpointElapsedSeconds),
      trainedSteps: cumulative(state.run.baseTrainedSteps, state.checkpointSave.steps) });
  state.models.unshift(result);
  state.latestSavedModelId = id;
  state.checkpointSave.phase = 'saved';
  state.checkpointSave.modelId = id;
  event(state, '快照已保存并验证', `固定在 ${state.checkpointSave.steps.toLocaleString('zh-CN')} 环境步，训练状态保持不变`);
}

export function transition(current, action, payload = {}) {
  const state = clone(current);
  state.notice = null;
  if (state.connection === 'disconnected' && !['reconnect', 'clear-notice'].includes(action)) {
    state.notice = '连接断开，先重新核对运行状态。训练是否继续不能由连接状态判断。';
    return state;
  }
  switch (action) {
    case 'start': {
      if (ACTIVE.has(state.phase)) { state.notice = '本轮任务仍在进行，不能重复开始。'; break; }
      const id = Object.hasOwn(payload, 'modelId') ? payload.modelId : state.selectedModelId;
      const selected = state.models.find(value => value.id === id);
      if (id && (!selected || !selected.compatible || !selected.verified)) { state.notice = '请选择兼容且已验证的模型，或从新策略开始。'; break; }
      const duration = payload.durationSeconds ?? 900;
      if (!Number.isFinite(duration) || duration < 0 || duration > 604800 || (duration > 0 && duration < 6)) { state.notice = '训练时长需为 0（不限时），或 6 秒至 7 天。'; break; }
      state.run.id = `demo-run-${String(state._clock.run++).padStart(3, '0')}`;
      state.run.name = `基线平衡 · ${state.run.id.slice(-3)}`;
      state.run.parentModelId = id || null;
      state.run.baseTrainedSeconds = id ? knownAmount(selected.trainedSeconds) : 0;
      state.run.baseTrainedSteps = id ? knownAmount(selected.trainedSteps) : 0;
      state.selectedModelId = id || null;
      state.run.steps = 0;
      state.run.elapsedSeconds = 0;
      state.run.schedule = duration ? 'timed' : 'unlimited';
      state.run.durationSeconds = duration;
      state.phase = 'preparing';
      state.error = null;
      state._clock.phase = 0;
      state._clock.evaluation = 0;
      state.evaluation.points = [];
      state.data.evaluationAt = null; state.data.evaluationSteps = null; state.data.evalCount = 0;
      state.preview = { status: 'off', modelId: null, runId: null, steps: null, syncedAt: null,
        sampleAt: null, telemetry: { episodeSeconds: null, resets: null, leftLeg: null,
          rightLeg: null, leftWheel: null, rightWheel: null }, error: null };
      state._clock.preview = 0;
      state.checkpointSave = { phase: 'idle', steps: null, requestedAt: null, modelId: null, error: null };
      state._clock.checkpointSave = 0;
      state._clock.checkpointElapsedSeconds = 0;
      state.preparation.elapsedSeconds = 0;
      state.preparation.stages = state.preparation.stages.map(value => ({ ...value,
        status: ['cached', 'passed'].includes(value.status) ? 'cached' : 'waiting', seconds: 0 }));
      event(state, '训练任务已创建', id ? '从所选模型继续；优化器重新初始化' : '从新策略开始');
      break;
    }
    case 'pause-training':
      if (state.phase !== 'training') { state.notice = '只有训练进行中可以暂停训练。'; break; }
      state.phase = 'paused';
      event(state, '训练已暂停', '冻结训练计时、采样与评估，预览可独立运行');
      break;
    case 'resume-training':
      if (state.phase !== 'paused') { state.notice = '当前训练没有暂停，不会重复开始任务。'; break; }
      state.phase = 'training';
      event(state, '训练已恢复', '继续同一运行的计时、采样与评估');
      break;
    case 'save-snapshot':
      if (!['training', 'paused'].includes(state.phase)) { state.notice = '仅运行中或暂停的训练可以保存快照；停止保存流程不能同时执行。'; break; }
      if (checkpointBusy(state)) { state.notice = '快照正在保存与校验，请等待完成。'; break; }
      requestCheckpoint(state);
      break;
    case 'stop':
      if (checkpointBusy(state)) {
        state.notice = '快照正在保存与校验，完成后可以停止并保存。';
      } else if (state.phase === 'preparing') {
        state.phase = 'cancelled';
        state.preparation.stages = state.preparation.stages.map(value => ['running', 'waiting'].includes(value.status) ? { ...value, status: 'cancelled' } : value);
        event(state, '准备已取消', '已通过或已复用阶段仍保留');
      } else if (['training', 'paused', 'compiling'].includes(state.phase)) {
        state.phase = 'stopping'; state._clock.phase = 0;
        event(state, '已请求停止', '等待当前计算段结束，再保存和校验');
      } else state.notice = ACTIVE.has(state.phase) ? '停止与保存正在进行，请等待校验。' : '当前没有运行中的训练任务。';
      break;
    case 'select-model': {
      if (ACTIVE.has(state.phase)) { state.notice = '运行期间来源锁定；可查看模型信息。'; break; }
      const selected = state.models.find(value => value.id === payload.modelId);
      if (payload.modelId && (!selected || !selected.compatible || !selected.verified)) { state.notice = selected?.reason || '模型尚未验证或不兼容。'; break; }
      state.selectedModelId = payload.modelId || null;
      state.notice = '已选择下一轮模型，尚未开始训练。';
      break;
    }
    case 'preview-open':
      if (['syncing', 'running', 'paused', 'ended'].includes(state.preview.status)) break;
      if (!state.latestSavedModelId && !['training', 'paused', 'stopping', 'saving', 'exporting', 'verifying'].includes(state.phase)) { state.notice = '尚无可预览的模型。'; break; }
      state.preview.status = 'syncing'; state.preview.error = null; state._clock.preview = 0;
      event(state, '请求预览策略', '仅同步一次，观看期间保持此版本');
      break;
    case 'preview-close': state.preview.status = 'off'; state._clock.preview = 0; break;
    case 'preview-toggle-pause':
      if (state.preview.status === 'running') state.preview.status = 'paused';
      else if (state.preview.status === 'paused') state.preview.status = 'running';
      else state.notice = '请先开启策略预览。';
      break;
    case 'preview-reset':
      if (!['running', 'paused', 'ended'].includes(state.preview.status)) { state.notice = '画面关闭时不能重置回合。'; break; }
      state.preview.telemetry.episodeSeconds = 0;
      state.preview.telemetry.resets = (state.preview.telemetry.resets || 0) + 1;
      state.preview.sampleAt = state.now;
      if (state.preview.status === 'ended') state.preview.status = 'running';
      break;
    case 'reconnect':
      state.connection = 'connected';
      state.data.receivedAt = state.now;
      state.data.progressAt = state.now;
      state.notice = '已重新核对当前运行；模型来源和未提交配置保持不变。';
      event(state, '连接已恢复', '重新读取运行身份与阶段，没有创建新任务');
      break;
    case 'clear-notice': state.notice = null; break;
    default: state.notice = '当前操作不可用。';
  }
  return state;
}

function evaluate(state) {
  const previous = state.evaluation.points.at(-1);
  const count = state.evaluation.points.length;
  const seconds = round(Math.min(0.95, (previous?.episodeSeconds ?? 0.22) + 0.025 + Math.sin(count) * 0.018));
  state.evaluation.points.push({ steps: state.run.steps, episodeSeconds: seconds,
    reward: round((previous?.reward ?? 10) + 2.2 + Math.cos(count) * 1.1, 2),
    valueLoss: round(Math.max(0.1, 0.42 - count * 0.009)), policyLoss: round(-0.004 + Math.sin(count) * 0.002, 5), at: state.now });
  state.evaluation.points = state.evaluation.points.slice(-24);
  state.data.evaluationAt = state.now; state.data.evaluationSteps = state.run.steps;
  state.data.evalCount += 1;
}

function tickOnce(state) {
  state.now += 1000;
  if (state.connection === 'disconnected') return;
  state.data.receivedAt = state.now;
  if (state.connection !== 'stale' && ACTIVE.has(state.phase) && state.phase !== 'paused') state.data.progressAt = state.now;
  if (state.connection === 'stale') return;
  if (ACTIVE.has(state.phase)) state._clock.phase += 1;
  if (state.phase === 'preparing') {
    state.preparation.elapsedSeconds += 1;
    const running = state.preparation.stages.find(value => value.status === 'running');
    if (running) { running.seconds += 1; if (state._clock.phase >= 4) { running.status = 'passed'; running.detail = '本轮验证通过 · 检查记录已保留'; state._clock.phase = 0; event(state, `${running.label}已通过`, '阶段结果立即保留'); } }
    const pending = state.preparation.stages.find(value => ['waiting', 'failed', 'cancelled'].includes(value.status));
    if (!state.preparation.stages.some(value => value.status === 'running')) {
      if (pending) { pending.status = 'running'; state._clock.phase = 0; }
      else if (state._clock.phase >= 2) { state.phase = 'compiling'; state._clock.phase = 0; event(state, '准备检查完成', '训练计时开始，包含编译与评估'); }
    }
  } else if (state.phase === 'compiling') {
    state.run.elapsedSeconds += 1;
    if (state._clock.phase >= 4) { state.phase = 'training'; state._clock.phase = 0; event(state, '开始训练', '当前进度与评估记录分开更新'); }
  } else if (state.phase === 'training') {
    state.run.elapsedSeconds += 1; state.run.steps += 8192; state._clock.evaluation += 1;
    if (state._clock.evaluation >= 45) { evaluate(state); state._clock.evaluation = 0; }
    if (state.run.durationSeconds && state.run.elapsedSeconds >= state.run.durationSeconds && !checkpointBusy(state)) { state.phase = 'stopping'; state._clock.phase = 0; event(state, '已到训练时限', '等待安全计算边界后保存'); }
  } else if (SAVING.includes(state.phase) && state._clock.phase >= 2) {
    const index = SAVING.indexOf(state.phase);
    if (index === SAVING.length - 1) complete(state);
    else { state.phase = SAVING[index + 1]; event(state, { saving: '正在保存参数', exporting: '正在导出模型', verifying: '正在校验模型' }[state.phase], '完成后自动进入下一阶段'); }
    state._clock.phase = 0;
  }
  advanceCheckpoint(state);
  if (state.preview.status === 'syncing') {
    state._clock.preview += 1;
    if (state._clock.preview >= 2) { snapshot(state, 'running'); event(state, '预览快照已同步', `固定在 ${state.preview.steps.toLocaleString('zh-CN')} 环境步`); }
  } else if (state.preview.status === 'running') {
    state.preview.sampleAt = state.now;
    state.preview.telemetry.episodeSeconds = round((state.preview.telemetry.episodeSeconds + 0.03) % 0.6);
  }
}

export function tick(current, seconds = 1) {
  const state = clone(current);
  const count = Number.isFinite(seconds) ? Math.max(0, Math.min(3600, Math.floor(seconds))) : 0;
  for (let i = 0; i < count; i += 1) tickOnce(state);
  return state;
}
