import { ACTIVE_PHASES, SCENARIOS, createDemo, transition, tick } from './mock-engine.mjs';
import { initWorkspaceLayout } from './layout.mjs';
import { createInteractiveChart } from './chart-view.mjs';
import { createCurveFixture } from './chart-model.mjs';
import { initCheckpointDirectory } from './checkpoint-directory.mjs';

// This page deliberately has no transport, persistence, or production adapter.
const $ = id => document.getElementById(id);
document.body.dataset.developer=String(new URLSearchParams(location.search).get('dev')==='1');
const activePhases = new Set(ACTIVE_PHASES);
const savingPhases = new Set(['stopping', 'saving', 'exporting', 'verifying']);
const validScenarios = new Set(SCENARIOS.map(item => item.id));
const initialScenario = new URLSearchParams(location.search).get('scene');
let state = createDemo(validScenarios.has(initialScenario) ? initialScenario : 'training');
let playing = false;
let draftDuration = state.run.durationSeconds || 0;
let lastChartSignature = '';
let lastChartPoints = null;
const chartViews = new Map();
const curveKinds = new Set(['dense-1k','dense-10k','wide-loss','signed-loss','tiny-changes','gaps','constant']);
const requestedCurve = new URLSearchParams(location.search).get('curve');
let curveMode = curveKinds.has(requestedCurve) ? requestedCurve : 'current';
let fixtureCache = null;
let directoryController = null;
function curvePoints() { if(curveMode==='current')return state.evaluation.points; if(!fixtureCache)fixtureCache=createCurveFixture(curveMode,{endStep:Math.max(573440,state.run.steps||0),now:state.now}); return fixtureCache; }
let lastPreparationSignature = '';
let lastEventsSignature = '';
let lastPhase = '';
let viewportCount = 1;
const windowFolds = new Map();
const checkpointAliases = new Map([['demo-model-023','C023'],['demo-model-019','C019'],['demo-author-policy','A000'],['demo-torque-policy','T001']]);
let nextCheckpointAlias = 24;
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
const disclosureMotions = new WeakMap();
const motionTextIds = new Set(['nav-checkpoint','phase-chip','viewport-status','draft-state']);
function animateNode(node, frames, options = {}) { return !motionPreference.matches && node?.animate ? node.animate(frames, {duration:180,easing:'cubic-bezier(.22,1,.36,1)',...options}) : null; }
function syncMotionPreference() { document.body.dataset.reducedMotion=String(motionPreference.matches); }
motionPreference.addEventListener('change', syncMotionPreference); syncMotionPreference();
function checkpointAlias(id) { if (!id) return '—'; if (!checkpointAliases.has(id)) checkpointAliases.set(id, `C${String(nextCheckpointAlias++).padStart(3,'0')}`); return checkpointAliases.get(id); }
const runAlias = () => `R${state.run.id.match(/(\d+)$/)?.[1] || '—'}`;

const number = (value, digits = 0) => Number.isFinite(value)
  ? value.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
const compactValue = value => Number.isFinite(value) ? String(Number(value.toPrecision(6))) : '—';
const time = value => Number.isFinite(value)
  ? new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Hong_Kong', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(value) : '尚无记录';
const clock = value => {
  const seconds = Math.max(0, Math.floor(value || 0));
  return `${seconds >= 3600 ? `${Math.floor(seconds / 3600)}:` : ''}${String(Math.floor(seconds % 3600 / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
};
const age = value => {
  if (!Number.isFinite(value)) return '等待首次记录';
  const seconds = Math.max(0, Math.floor((state.now - value) / 1000));
  if (seconds < 2) return '刚刚';
  if (seconds < 60) return `${seconds} 秒前`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒前`;
  return `${Math.floor(seconds / 3600)} 小时前`;
};
function text(id, value) { const node = $(id); const next = String(value ?? ''); if (node.textContent !== next) { node.textContent = next; if(motionTextIds.has(id)) animateNode(node,[{opacity:.35,transform:'translateY(2px)'},{opacity:1,transform:'translateY(0)'}]); } }
function hidden(id, value) { const node=$(id); const wasHidden=node.hidden; node.hidden=Boolean(value); if(wasHidden&&!node.hidden) animateNode(node,[{opacity:0,transform:'translateY(4px)'},{opacity:1,transform:'translateY(0)'}]); }
function element(tag, className, value) { const node = document.createElement(tag); if (className) node.className = className; if (value !== undefined) node.textContent = value; return node; }
function chip(id, label, tone = '') { text(id, label); const node=$(id); for(const name of [...node.classList]) if(name.startsWith('tone-'))node.classList.remove(name); node.classList.add('status-chip'); if(tone)node.classList.add(`tone-${tone}`); }
const currentModel = id => state.models.find(model => model.id === id);
const latestEvaluation = () => state.evaluation.points.filter(point => Number.isFinite(point.episodeSeconds)).at(-1);
const knownMode = () => state.run.mode.startsWith('hybrid_');
const canControl = () => state.connection === 'connected';
const isActive = () => activePhases.has(state.phase);

const phaseLabels = {
  idle: ['等待开始', 'neutral'], preparing: ['准备检查中', 'info'], compiling: ['编译中', 'info'],
  training: ['训练进行中', ''], stopping: ['等待安全停止', 'warn'], saving: ['保存参数中', 'info'],
  paused: ['训练已暂停', 'neutral'],
  exporting: ['导出模型中', 'info'], verifying: ['校验模型中', 'info'], completed: ['已完成 · 模型已验证', ''],
  cancelled: ['准备已取消', 'neutral'], failed: ['需要处理', 'error'],
};

function renderRun() {
  const run = state.run;
  const disconnected = state.connection === 'disconnected';
  const [label, tone] = phaseLabels[state.phase] || ['状态未知', 'neutral'];
  chip('phase-chip', disconnected ? '状态待核对' : label, disconnected ? 'error' : tone);
  text('run-name', run.name.replace(/\s*·\s*\d+$/, ''));
  text('nav-run-id', runAlias());
  text('nav-checkpoint', checkpointAlias(state.latestSavedModelId));
  $('nav-checkpoint').title = state.latestSavedModelId || '暂无已验证checkpoint';
  text('run-mode', `${run.modeLabel} · ${run.actionLabel}`);
  text('run-id', run.id.toUpperCase());
  text('run-steps', number(run.steps));
  const preparing = state.phase === 'preparing';
  const idle = state.phase === 'idle';
  const saving = savingPhases.has(state.phase);
  text('clock-label', disconnected ? '上次收到的训练计时' : preparing ? '准备已用时' : saving ? '训练计时已结束' : idle ? '训练预算' : isActive() && run.durationSeconds ? '剩余训练时间' : '训练已用时');
  text('run-clock', preparing ? clock(state.preparation.elapsedSeconds) : idle ? (draftDuration ? clock(draftDuration) : '不限时') : isActive() && !saving && run.durationSeconds ? clock(run.durationSeconds - run.elapsedSeconds) : clock(run.elapsedSeconds));

  const button = $('main-action');
  if (saving) { button.textContent = '正在完成保存'; button.className = 'button button-secondary'; button.disabled = true; text('action-hint', '校验通过后发布模型'); }
  else if (isActive()) { button.textContent = preparing ? '取消准备' : '停止并保存'; button.className = 'button button-stop'; button.disabled = !canControl(); text('action-hint', preparing ? '已通过的阶段仍保留' : '等待当前计算段结束后保存'); }
  else { button.textContent = state.phase === 'failed' || state.phase === 'cancelled' ? '重新准备' : state.selectedModelId ? '继续训练' : '开始训练'; button.className = 'button button-primary'; button.disabled = !canControl(); text('action-hint', '将在完成准备后开始训练'); }
  const showAlert = state.connection !== 'connected' || Boolean(state.error);
  hidden('connection-alert', !showAlert);
  hidden('reconnect-button', state.connection === 'connected');
  if (disconnected) {
    text('connection-alert-title', '连接已断开 · 当前训练状态待确认');
    text('connection-alert-description', '保留最后收到的数据。连接中断不能说明训练已停止，恢复后会核对同一运行。');
  } else if (state.connection === 'stale') {
    text('connection-alert-title', '页面已连接，进度数据暂未更新');
    text('connection-alert-description', '最后收到的数据已标注时间；新的评估尚未到达。');
  } else if (state.error) {
    text('connection-alert-title', state.scenario === 'save-failed' ? '本轮模型尚未通过校验' : knownMode() && !state.run.steps ? '准备检查未完成' : '任务需要处理');
    text('connection-alert-description', state.error);
  }
  hidden('action-notice', !state.notice);
  if (state.notice) text('action-notice', state.notice);
  document.body.dataset.phase = state.phase;
  document.body.dataset.connection = state.connection;
}

function renderProcess() {
  const groups = ['prepare', 'train', 'save', 'complete'];
  const stageIndex = state.phase === 'idle' ? -1 : ['preparing', 'cancelled'].includes(state.phase) || (state.phase === 'failed' && !state.run.steps) ? 0
    : ['compiling', 'training'].includes(state.phase) ? 1 : savingPhases.has(state.phase) ? 2
      : state.phase === 'completed' ? 3 : state.scenario === 'save-failed' ? 2 : 1;
  for (const [index, name] of groups.entries()) {
    const node = document.querySelector(`[data-stage="${name}"]`);
    node.className = index < stageIndex || (index === 3 && state.phase === 'completed') ? 'is-complete' : index === stageIndex ? 'is-active' : '';
    if (state.phase === 'failed' && index === stageIndex) node.classList.add('is-error');
    if (index === stageIndex) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current');
  }
  text('step-prepare-description', state.phase === 'idle' ? '等待开始' : state.phase === 'cancelled' ? '已通过阶段保留' : state.phase === 'preparing' ? '逐阶段检查' : stageIndex > 0 ? '已就绪' : '检查未完成');
  text('step-train-description', state.phase === 'paused' ? '暂停，进度保留' : state.phase === 'compiling' ? '计算图编译中' : state.phase === 'training' ? '正在采样与更新' : stageIndex > 1 ? '本轮计算结束' : '准备完成后开始');
  text('step-save-description', ({ stopping: '等待计算段结束', saving: '正在写入参数', exporting: '正在导出模型', verifying: '正在核对一致性', completed: '保存与校验通过' })[state.phase] || '等待训练结束');
  text('step-complete-description', state.phase === 'completed' ? '模型可用于下一轮' : '校验通过后可继续');
  const description = state.connection === 'disconnected' ? '显示断线前的阶段，恢复后重新核对' : state.phase === 'preparing' ? '准备检查单独计时，已通过阶段会保留'
    : state.phase === 'compiling' ? '本阶段完成时间未知，训练预算包含编译'
      : savingPhases.has(state.phase) ? '保存另计时间；校验通过才发布新模型'
        : state.phase === 'completed' ? '本轮保存与校验已完成'
          : state.phase === 'cancelled' ? '尚未进入正式训练，可继续未完成的准备'
            : '训练计时包含编译与评估';
  text('process-description', description);
  const progress = $('training-progress');
  if (['preparing', 'compiling', 'stopping', 'saving', 'exporting', 'verifying'].includes(state.phase) || (state.phase === 'training' && !state.run.durationSeconds)) {
    progress.removeAttribute('value');
    progress.setAttribute('aria-label', `${phaseLabels[state.phase][0]}，完成时间未知`);
  } else {
    progress.max = state.run.durationSeconds || 1;
    progress.value = Math.min(state.run.elapsedSeconds, progress.max);
    progress.setAttribute('aria-label', '训练时间预算进度，不代表模型效果');
  }
  hidden('preparation-details', !knownMode());
  if (knownMode()) {
    const done = state.preparation.stages.filter(stage => ['cached', 'passed'].includes(stage.status)).length;
    text('preparation-summary', `${done} / 3 项已就绪`);
    const signature = JSON.stringify(state.preparation.stages);
    if (signature !== lastPreparationSignature) {
      const states = { cached: '已复用', passed: '已通过', running: '检查中', waiting: '待上游完成', cancelled: '已取消', failed: '未通过' };
      $('preparation-stages').replaceChildren(...state.preparation.stages.map(stage => {
        const row = element('li'); const name = element('div'); name.append(element('span', '', stage.label), element('small', '', stage.detail));
        const status = element('span', 'stage-result', `${states[stage.status]}${stage.seconds ? ` · ${stage.seconds}秒` : ''}`); status.dataset.status = stage.status;
        row.append(name, status); return row;
      }));
      lastPreparationSignature = signature;
    }
  }
  if (lastPhase !== state.phase) {
    if (state.phase === 'preparing' && knownMode()) $('preparation-details').open = true;
    lastPhase = state.phase;
  }
}

function renderEvaluation() {
  const points=curvePoints();
  const validEpisodes=points.filter(point=>Number.isFinite(point.episodeSeconds));
  const latest = validEpisodes.at(-1);
  const previous = validEpisodes.at(-2);
  text('evaluation-context', curveMode!=='current' ? `测试 · ${number(points.length)} 点` : latest ? `${number(state.data.evaluationSteps)} 步 · ${age(state.data.evaluationAt)}` : '首次评估完成后更新');
  text('metric-duration', latest ? number(latest.episodeSeconds, 2) : '—');
  text('metric-reward', latest ? number(latest.reward, 1) : '—');
  text('metric-samples', latest ? number(state.data.evalEnvs) : '—');
  text('metric-points', `${curveMode==='current'?state.data.evalCount:points.length} 个记录点`);
  const lastLoss = points.at(-1);
  text('value-latest', compactValue(lastLoss?.valueLoss));
  text('policy-latest', compactValue(lastLoss?.policyLoss));
  text('chart-eval-step', latest ? `${number(latest.steps)} 步` : '等待首次评估');
  const change = (id, value, unit, suffix) => {
    text(id, Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${number(value, 2)}${unit} ${suffix}` : '等待第二次评估');
    $(id).className = `metric-note ${Number.isFinite(value) ? value > 0 ? 'is-positive' : value < 0 ? 'is-negative' : '' : ''}`;
  };
  change('duration-change', previous && latest ? latest.episodeSeconds - previous.episodeSeconds : null, '秒', '较上次');
  change('reward-change', previous && latest ? latest.reward - previous.reward : null, '', '较上次评估');
  if(points===lastChartPoints)return;
  const signature = JSON.stringify(points);
  if (signature !== lastChartSignature) {
    renderChart('duration-chart', 'episodeSeconds', '回合时长', 'duration');
    renderChart('reward-chart', 'reward', '回合奖励', 'reward');
    renderChart('value-chart', 'valueLoss', 'Value Loss', 'loss');
    renderChart('policy-chart', 'policyLoss', 'Policy Loss', 'policy');
    lastChartSignature = signature;
  }
  lastChartPoints=points;
}

function renderChart(id, key, label, series) {
  if(!chartViews.has(id))chartViews.set(id,createInteractiveChart($(id),{key,label,loss:key.endsWith('Loss'),series}));
  chartViews.get(id).setData(curvePoints());
}

function renderModels() {
  const source = currentModel(state.selectedModelId);
  const saved = currentModel(state.latestSavedModelId);
  text('source-model-name', source?.name || '从新策略开始');
  text('source-alias', checkpointAlias(state.selectedModelId));
  text('source-model-detail', source ? `${source.controlHz} Hz · ${Number.isFinite(source.trainedSeconds) ? `累计 ${clock(source.trainedSeconds)}` : '历史时长未记录'} · ${source.steps ? `${number(source.steps)} 步` : '步数未记录'}` : '尚未选择历史模型');
  text('source-model-state', source ? '数值校验通过' : '新策略 · 无父模型');
  text('training-model-name', state.phase === 'idle' ? '尚未开始下一轮' : state.run.name);
  text('training-model-steps', state.phase === 'idle' ? '配置尚未提交' : `${number(state.run.steps)} 步 · ${state.connection === 'disconnected' ? '断线前记录' : '本轮进度'}`);
  text('saved-model-name', saved ? `${checkpointAlias(saved.id)} · ${saved.name}` : '暂无已验证模型');
  text('saved-model-steps', saved ? `${Number.isFinite(saved.trainedSeconds) ? `累计 ${clock(saved.trainedSeconds)}` : '历史时长未记录'} · ${number(saved.trainedSteps ?? saved.steps)} 步` : '保存与校验完成后显示');
  text('view-model-name', state.preview.modelId ? state.preview.status === 'off' ? '上次同步的固定版本' : '打开时同步的固定版本' : '尚未同步');
  text('view-model-steps', state.preview.modelId ? `${number(state.preview.steps)} 步 · ${time(state.preview.syncedAt)}` : '每次打开时同步一次');
  text('choose-model', isActive() ? '查看可用来源' : '选择模型');
  text('model-lock-note', isActive() ? '任务进行中，来源已锁定；仍可查看模型信息。' : '选择只更新下一轮配置，点击开始后创建任务。');
  text('settings-state', '下一轮草稿');
  text('setting-action', state.run.actionLabel);
  text('setting-frequency', `${draftMode().controlHz} Hz`);
  text('setting-schedule', isActive() || state.phase === 'completed' ? state.run.durationSeconds ? `定时 · ${number(state.run.durationSeconds / 60)} 分钟` : '不限时' : draftDuration ? `定时 · ${draftDuration / 60} 分钟` : '不限时');
  text('setting-observations', `${state.run.obsDim} 维`);
  $('duration-select').disabled = !canControl();
  directoryController?.update();
}

function renderFreshness() {
  const labels = { connected: '连接正常', disconnected: '连接中断', stale: '数据延迟' };
  text('connection-chip', labels[state.connection]);
  $('connection-chip').className = `connection-chip ${state.connection !== 'connected' ? `is-${state.connection}` : ''}`;
  text('received-age', age(state.data.receivedAt));
  text('progress-age', age(state.data.progressAt));
  text('evaluation-age', age(state.data.evaluationAt));
  for (const [id, at] of [['received-age', state.data.receivedAt], ['progress-age', state.data.progressAt], ['evaluation-age', state.data.evaluationAt]]) $(id).title = time(at);
  text('data-lag-note', state.connection === 'disconnected' ? '当前展示断线前的数据，训练是否继续需恢复后核对。'
    : state.connection === 'stale' ? '读取请求仍正常，训练进度与评估记录尚未刷新。'
      : !state.data.evaluationAt ? '尚无本轮评估记录；不会使用旧运行的数据填充。'
        : `训练进度比最近评估多 ${number(Math.max(0, state.run.steps - state.data.evaluationSteps))} 步，等待下一次评估。`);
}

function renderPreview() {
  const preview = state.preview;
  const active = ['syncing', 'running', 'paused', 'ended'].includes(preview.status);
  const labels = { off: ['已关闭', 'neutral'], syncing: ['同步中', 'info'], running: ['预览已开启', ''], paused: ['已暂停', 'warn'], failed: ['同步未完成', 'error'], ended: ['回合已结束', 'neutral'] };
  chip('preview-chip', ...(labels[preview.status] || labels.off));
  text('preview-description', preview.status === 'failed' ? preview.error : preview.status === 'syncing' ? '正在准备固定策略快照；训练继续进行，可取消同步。'
    : preview.status === 'off' ? '开启时同步一次，观看期间保持同一策略版本。'
      : preview.status === 'paused' ? '预览已暂停，保留下方最后一次采样；训练继续进行。'
        : preview.status === 'ended' ? '本次回合结束，可重置回合。训练状态不受影响。' : '策略版本保持固定，训练与评估记录继续更新。');
  text('preview-version', preview.modelId ? `${number(preview.steps)} 步 · ${time(preview.syncedAt)} 同步 · ${checkpointAlias(preview.modelId)}` : '尚未同步策略');
  text('preview-open', preview.status === 'syncing' ? '取消同步' : active ? '关闭预览' : preview.status === 'failed' ? '重新同步' : '开启预览');
  $('preview-open').disabled = !canControl() || (!state.latestSavedModelId && !['training', 'paused', 'stopping', 'saving', 'exporting', 'verifying'].includes(state.phase));
  hidden('preview-pause', !['running', 'paused'].includes(preview.status));
  hidden('preview-reset', !['running', 'paused', 'ended'].includes(preview.status));
  text('preview-pause', preview.status === 'paused' ? '继续预览' : '暂停预览');
  $('preview-pause').disabled = !canControl();
  $('preview-reset').disabled = !canControl();
  hidden('telemetry-panel', preview.sampleAt === null);
  const stale = preview.status !== 'running' || state.connection !== 'connected';
  text('telemetry-state', preview.sampleAt === null ? '未采样' : stale ? '上次采样' : '实时遥测');
  text('telemetry-time', `${time(preview.sampleAt)} · ${age(preview.sampleAt)}`);
  const values = [
    ['左腿输入', preview.telemetry.leftLeg, knownMode() ? 'N·m' : 'rad'],
    ['右腿输入', preview.telemetry.rightLeg, knownMode() ? 'N·m' : 'rad'],
    ['左轮输入', preview.telemetry.leftWheel, knownMode() ? 'N·m' : 'rad/s'],
    ['右轮输入', preview.telemetry.rightWheel, knownMode() ? 'N·m' : 'rad/s'],
  ];
  if ($('telemetry-values').childElementCount !== values.length) $('telemetry-values').replaceChildren(...values.map(() => { const row = element('div'); row.append(element('dt'), element('dd', 'numeric')); return row; }));
  for (const [index, [label, value, unit]] of values.entries()) { const row = $('telemetry-values').children[index]; row.children[0].textContent = label; row.children[1].textContent = `${number(value, 4)} ${unit}`; }
  document.body.dataset.previewStatus = preview.status;
  text('preview-short-version', preview.modelId ? `${number(preview.steps)} 步 · ${time(preview.syncedAt)}` : '尚未同步');
  const viewportLabel = ({ off:'预览未开启', syncing:'正在同步策略', running:'等待画面', paused:'预览已暂停', failed:'同步未完成', ended:'回合已结束' })[preview.status] || '预览未开启';
  text('viewport-status', viewportLabel);
  for (const tile of $('viewport-grid').children) {
    const title = tile.querySelector('.viewport-empty-title');
    const detail = tile.querySelector('.viewport-empty-detail');
    title.textContent = viewportLabel;
    detail.textContent = preview.modelId ? `${number(preview.steps)} 环境步策略` : '选择模型后开启预览';
  }
}

function renderViewportLayout() {
  for (const tile of $('viewport-grid').children) windowFolds.set(tile.dataset.window, tile.open);
  $('viewport-grid').dataset.count = String(viewportCount);
  $('viewport-grid').replaceChildren(...Array.from({length:viewportCount}, (_, index) => {
    const id = String(index + 1).padStart(2,'0');
    const tile = element('details','viewport-tile'); tile.dataset.window = id; tile.open = windowFolds.get(id) ?? true;
    const summary = element('summary'); summary.append(element('span','disclosure-label',viewportCount > 1 ? id : '展开')); summary.setAttribute('aria-label',`折叠或展开${viewportCount > 1 ? `窗口 ${id}` : '当前窗口'}`);
    const body = element('div','fold-body'); body.append(element('p','viewport-empty-title','预览未开启'), element('p','viewport-empty-detail',''));
    tile.append(summary,body); return tile;
  }));
  text('window-layout-label', viewportCount === 1 ? '单窗口' : `${Math.sqrt(viewportCount)} × ${Math.sqrt(viewportCount)} 布局`);
  [...$('viewport-grid').children].forEach((tile,index)=>animateNode(tile,[{opacity:0,transform:'translateY(5px) scale(.99)'},{opacity:1,transform:'translateY(0) scale(1)'}],{duration:220,delay:index*18}));
}

function renderTransport() {
  const source = currentModel(state.selectedModelId);
  const idle = state.phase === 'idle';
  const baseTime = idle ? source ? source.trainedSeconds : 0 : state.run.baseTrainedSeconds;
  const baseSteps = idle ? source ? source.trainedSteps : 0 : state.run.baseTrainedSteps;
  const elapsed = idle ? 0 : state.run.elapsedSeconds;
  const budget = idle ? draftDuration : state.run.durationSeconds;
  const knownHistory = Number.isFinite(baseTime);
  const totalElapsed = knownHistory ? baseTime + elapsed : elapsed;
  const totalTarget = budget ? (knownHistory ? baseTime : 0) + budget : 0;
  const progress = $('training-progress');
  if (totalTarget) { progress.max = totalTarget; progress.value = Math.min(totalTarget,totalElapsed); }
  else progress.removeAttribute('value');
  const fillRatio = totalTarget ? Math.max(0,Math.min(1,totalElapsed/totalTarget)) : 1;
  $('training-progress-track').style.setProperty('--progress-fill', `${fillRatio*100}%`);
  progress.setAttribute('aria-label', knownHistory ? '累计训练时长进度，包含checkpoint历史' : '本轮时长进度，历史时长未记录');
  text('clock-label', knownHistory ? '累计 / 计划时长' : '本轮 / 新增预算');
  text('run-clock', `${clock(totalElapsed)} / ${totalTarget ? clock(totalTarget) : '不限时'}`);
  text('run-steps', number(Number.isFinite(baseSteps) ? baseSteps + state.run.steps : state.run.steps));
  text('steps-label', Number.isFinite(baseSteps) ? '累计环境步' : '本轮环境步');
  const percent = totalTarget ? `${Math.min(100,totalElapsed/totalTarget*100).toFixed(1).replace(/\.0$/,'')}%` : '不限时';
  text('transport-summary', `${phaseLabels[state.phase]?.[0]} · ${knownHistory ? '累计' : '本轮'} ${clock(totalElapsed)} · ${percent}`);
  const saving = state.checkpointSave?.phase;
  document.body.dataset.saveStatus=saving || 'idle';
  const saveState = saving === 'saving' ? '快照保存中' : saving === 'verifying' ? '快照校验中' : saving === 'saved' ? `已保存 ${checkpointAlias(state.checkpointSave.modelId)}` : '';
  text('transport-status', `${state.connection === 'connected' ? phaseLabels[state.phase]?.[0] : '状态待核对'}${saveState ? ` · ${saveState}` : ''}`);
  text('transport-percent', percent);
  text('transport-detail', `${knownHistory ? `历史 ${clock(baseTime)} + 本轮 ${clock(elapsed)}` : '历史时长未记录，仅显示本轮'} · 追加预算 ${budget ? clock(budget) : '不限时'}${saveState ? ` · ${saveState}` : ''}`);
  progress.title = $('transport-detail').textContent;
  const paused = state.phase === 'paused';
  const saveBusy = ['saving','verifying'].includes(saving);
  $('training-start').textContent = paused ? '继续' : '开始';
  $('training-start').disabled = !canControl() || (isActive() && !paused);
  $('training-pause').disabled = !canControl() || state.phase !== 'training';
  $('training-save').textContent = saveBusy ? saving === 'saving' ? '保存中' : '校验中' : '保存';
  $('training-save').disabled = !canControl() || !['training','paused'].includes(state.phase) || saveBusy;
}

function renderEvents() {
  const events = state.events.slice(0, 5);
  const signature = JSON.stringify(events);
  if (signature !== lastEventsSignature) {
    $('event-list').replaceChildren(...(events.length ? events.map(item => {
      const row = element('li'); const stamp = element('time', '', time(item.at)); stamp.dateTime = new Date(item.at).toISOString();
      const detail = element('div'); detail.append(element('strong', '', item.title), element('p', '', item.detail)); row.append(stamp, detail); return row;
    }) : [element('li', 'caption', '开始任务后显示运行记录。')]));
    lastEventsSignature = signature;
  }
}

function render() {
  renderRun(); renderProcess(); renderEvaluation(); renderModels(); renderFreshness(); renderPreview(); renderEvents();
  renderTransport();
  text('demo-play', playing ? '暂停' : '播放');
  $('demo-play').setAttribute('aria-label', playing ? '暂停演示计时' : '播放演示计时');
  $('demo-play').setAttribute('aria-pressed', String(playing));
  if ($('state-inspector').closest('details').open) text('state-inspector', JSON.stringify({ demo: true, scenario: state.scenario, phase: state.phase, connection: state.connection, run: state.run, evaluation: state.data, selectedModelId: state.selectedModelId, latestSavedModelId: state.latestSavedModelId, preview: { status: state.preview.status, steps: state.preview.steps, syncedAt: state.preview.syncedAt }, checkpointSave:state.checkpointSave, clockPlaying: playing }, null, 2));
}

function act(action, payload) { state = transition(state, action, payload); render(); }
function play() { playing = !playing; render(); }
function chooseScenario(id) {
  state = createDemo(id); playing = false; draftDuration = state.run.durationSeconds || 0;
  $('duration-select').value = String(draftDuration);
  syncDraftFromRun();
  lastChartSignature = ''; lastChartPoints=null; fixtureCache=null; lastPreparationSignature = ''; lastEventsSignature = ''; lastPhase = '';
  const url = new URL(location.href); url.searchParams.set('scene', state.scenario); history.replaceState(null, '', url);
  render();
}

function renderModelOptions() {
  const locked = isActive() || !canControl();
  text('model-dialog-description', locked ? '本轮任务来源已锁定；以下信息可查看，结束后才能改变下一轮来源。' : '选择下一轮使用的模型，然后点击开始。');
  const options = [{ id: '', name: '从新策略开始', steps: null, compatible: true, verified: true, reason: '不加载历史模型', sourceLabel: '新训练' }, ...state.models];
  $('model-options').replaceChildren(...options.map(model => {
    const chosen = (state.selectedModelId || '') === model.id;
    const button = element('button', `model-option${chosen ? ' is-selected' : ''}`);
    button.type = 'button'; button.dataset.modelId = model.id;
    button.disabled = locked || !model.compatible || !model.verified;
    const heading = element('strong', '', model.name); heading.append(element('span', 'todo-tag', chosen ? '已选来源' : !model.compatible ? '不兼容' : '可选'));
    const meta = element('span', '', model.steps === null ? '从初始策略开始' : `${model.controlHz} Hz · ${model.steps ? `${number(model.steps)} 步` : '训练步数未记录'} · ${time(model.savedAt)}`);
    const note = element('small', '', !model.compatible ? model.reason : model.sourceLabel);
    button.append(heading, meta, note); return button;
  }));
}

$('scenario-select').replaceChildren(...SCENARIOS.map(scenario => { const option = element('option', '', scenario.label); option.value = scenario.id; return option; }));
$('scenario-select').value = state.scenario;
$('scenario-select').addEventListener('change', event => chooseScenario(event.target.value));
$('demo-reset').addEventListener('click', () => chooseScenario($('scenario-select').value));
$('demo-play').addEventListener('click', play);
function performPrimaryAction() {
  if (isActive()) act('stop'); else act('start', { modelId: state.selectedModelId, durationSeconds: draftDuration });
  if (canControl()) playing = true;
  render();
}
$('main-action').addEventListener('click', performPrimaryAction);
$('duration-select').addEventListener('change', event => { draftDuration = Number(event.target.value); render(); });
$('reconnect-button').addEventListener('click', () => act('reconnect'));
$('preview-open').addEventListener('click', () => {
  if (['syncing', 'running', 'paused', 'ended'].includes(state.preview.status)) act('preview-close');
  else { act('preview-open'); if (state.preview.status === 'syncing') { playing = true; render(); } }
});
$('preview-pause').addEventListener('click', () => act('preview-toggle-pause'));
$('preview-reset').addEventListener('click', () => act('preview-reset'));
function showDialog(dialog) { if(dialog.open)return; dialog.showModal(); animateNode(dialog,[{opacity:0,transform:'translateY(8px) scale(.99)'},{opacity:1,transform:'translateY(0) scale(1)'}],{duration:200}); }
async function closeDialog(dialog) { if(!dialog.open || dialog.dataset.closing)return; dialog.dataset.closing='true'; const motion=animateNode(dialog,[{opacity:1,transform:'translateY(0)'},{opacity:0,transform:'translateY(5px)'}],{duration:130}); if(motion)await motion.finished.catch(()=>{}); dialog.close(); delete dialog.dataset.closing; }
$('choose-model').addEventListener('click', () => { renderModelOptions(); showDialog($('model-dialog')); });
$('model-options').addEventListener('click', event => { const button = event.target.closest('button[data-model-id]'); if (button && !button.disabled) { act('select-model', { modelId: button.dataset.modelId || null }); closeDialog($('model-dialog')); } });
for (const button of document.querySelectorAll('[data-close-dialog]')) button.addEventListener('click', () => closeDialog($(button.dataset.closeDialog)));
for (const button of document.querySelectorAll('[data-open-backlog]')) button.addEventListener('click', () => showDialog($('backlog-dialog')));
for (const dialog of document.querySelectorAll('dialog')) {
  dialog.addEventListener('cancel',event=>{event.preventDefault();closeDialog(dialog);});
  dialog.addEventListener('click', event => { if (event.target !== dialog) return; const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDialog(dialog); });
}

async function toggleDisclosure(details) {
  const previous=disclosureMotions.get(details);
  const opening=!(previous?.opening ?? details.open);
  const startHeight=details.getBoundingClientRect().height;
  if(previous) { previous.animation?.cancel(); details.style.overflow=previous.overflow; details.style.flex=previous.flex; }
  if(motionPreference.matches) { disclosureMotions.delete(details); details.open=opening; return; }
  const body=[...details.children].find(node=>node.tagName!=='SUMMARY');
  if(details.hasAttribute('data-pane')) {
    if(opening) details.open=true;
    const animation=animateNode(body,opening?[{opacity:0},{opacity:1}]:[{opacity:1},{opacity:0}],{duration:opening?210:100});
    const record={opening,animation,overflow:details.style.overflow,flex:details.style.flex}; disclosureMotions.set(details,record);
    if(animation)await animation.finished.catch(()=>{});
    if(disclosureMotions.get(details)===record) { details.open=opening; disclosureMotions.delete(details); }
    return;
  }
  const overflow=details.style.overflow, flex=details.style.flex;
  if(opening) { details.open=true; if(details.classList.contains('state-inspector'))render(); }
  const summary=details.querySelector(':scope > summary');
  const targetHeight=opening?details.getBoundingClientRect().height:summary.getBoundingClientRect().height;
  details.style.overflow='hidden';
  if(getComputedStyle(details).flexGrow!=='0')details.style.flex='0 0 auto';
  const animation=animateNode(details,[{height:`${startHeight}px`},{height:`${targetHeight}px`}],{duration:240,fill:'forwards'});
  const record={opening,animation,overflow,flex}; disclosureMotions.set(details,record);
  if(animation)await animation.finished.catch(()=>{});
  if(disclosureMotions.get(details)===record) { details.open=opening; animation?.cancel(); details.style.overflow=overflow; details.style.flex=flex; disclosureMotions.delete(details); }
}
document.addEventListener('click',event=>{
  const summary=event.target.closest('summary');
  if(!summary || event.target.closest('button,a,input,select'))return;
  const details=summary.parentElement;
  if(details.tagName==='DETAILS') { event.preventDefault(); toggleDisclosure(details); }
});
document.querySelector('.state-inspector').addEventListener('toggle', render);
document.addEventListener('submit', event => event.preventDefault());
const draftIds = ['param-name','duration-select','param-envs','param-seed','param-mode','param-hz','param-episode','param-lr','param-rollout','param-epochs','param-discount','param-gae','param-clip','param-eval-envs','param-eval-interval','param-fps'];
function readDraft() { return Object.fromEntries(draftIds.map(id => [id.replace('param-',''), $(id).type === 'number' || ['duration-select','param-hz'].includes(id) ? Number($(id).value) : $(id).value])); }
function draftMode() { const kind=$('param-mode').value; const hz=Number($('param-hz').value); return { mode:kind==='hybrid'?`hybrid_virtual_leg_force_${hz}`:kind, controlHz:kind==='original'?100:kind==='torque200'?200:hz, actionLabel:kind==='original'?'腿位置＋轮速度':kind==='torque200'?'全扭矩':'腿力＋轮扭矩', obsDim:kind==='original'?26:24 }; }
function syncDraftFromRun() {
  $('param-name').value = state.run.name.replace(/\s*·\s*\d+$/, '');
  $('param-mode').value = knownMode() ? 'hybrid' : state.run.mode === 'torque200' ? 'torque200' : 'original';
  $('param-hz').value = String(state.run.controlHz);
  $('param-hz').disabled = $('param-mode').value !== 'hybrid';
  $('duration-select').value = String(state.run.durationSeconds || 0);
  text('draft-state','下一轮草稿');
}
for (const id of draftIds) $(id).addEventListener('change', () => {
  if (id==='param-mode') { $('param-hz').value=$('param-mode').value==='torque200'?'200':'100'; $('param-hz').disabled=$('param-mode').value!=='hybrid'; }
  text('draft-state','草稿已修改');
  animateNode($(id),[{backgroundColor:'#2b2b2b'},{backgroundColor:'#101010'}],{duration:260});
  text('setting-frequency',`${draftMode().controlHz} Hz`);
  draftDuration=Number($('duration-select').value);
  renderTransport();
});
$('viewport-count').addEventListener('change', event => { viewportCount=Number(event.target.value); renderViewportLayout(); renderPreview(); });
$('training-start').addEventListener('click', () => {
  if (state.phase==='paused') { act('resume-training'); playing=true; render(); return; }
  for (const id of draftIds) { const input=$(id); if (!input.checkValidity()) { for (let parent=input.parentElement;parent;parent=parent.parentElement) if (parent.tagName==='DETAILS') parent.open=true; input.reportValidity(); return; } }
  const config=draftMode(); const selected=currentModel(state.selectedModelId);
  if (selected && (selected.mode!==config.mode || selected.controlHz!==config.controlHz)) { state={...state,notice:'来源与参数草稿不兼容，请先在模型来源中选择新策略或兼容模型。'}; render(); return; }
  const next=transition(state,'start',{modelId:state.selectedModelId,durationSeconds:draftDuration});
  if(next.run.id!==state.run.id) { Object.assign(next.run,config,{modeLabel:`${config.actionLabel} · ${config.controlHz} Hz`,name:`${$('param-name').value || '未命名任务'} · ${next.run.id.match(/\d+$/)?.[0] || ''}`,demoParameters:readDraft()}); }
  state=next; playing=true; render();
});
$('training-pause').addEventListener('click', () => act('pause-training'));
$('training-save').addEventListener('click', () => { act('save-snapshot'); playing=true; render(); });
const workspaceLayout=initWorkspaceLayout($('workbench'));
$('layout-reset').addEventListener('click', () => workspaceLayout.reset());
renderViewportLayout(); syncDraftFromRun();
$('curve-fixture').value=curveMode;
$('curve-fixture').addEventListener('change',event=>{curveMode=event.target.value;fixtureCache=null;lastChartPoints=null;lastChartSignature='';const url=new URL(location.href);if(curveMode==='current')url.searchParams.delete('curve');else url.searchParams.set('curve',curveMode);history.replaceState(null,'',url);render();});
directoryController=initCheckpointDirectory({getCheckpoint:()=>{const model=currentModel(state.latestSavedModelId);return model?{...model,alias:checkpointAlias(model.id)}:null;},onNotice:notice=>{state={...state,notice};render();},onDraftChange:()=>text('draft-state','目录草稿已修改'),showDialog,closeDialog});
setInterval(() => { if (playing) { state = tick(state); render(); } }, 1000);
render();
