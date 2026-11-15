import { buildWorkbenchView, commandRequest, checkpointLabel, checkpointTitle, formatNumber, formatSeconds, loadingView, metricRecords, previewImageFailure, previewRequestMode, previewStreamPlan, runKey, validateParameterDraft } from './adapter.mjs';
import { createWorkbenchTransport } from './transport.mjs';
import { initWorkspaceLayout } from './layout.mjs';
import { createInteractiveChart } from './chart-view.mjs';
import { formatChartValue } from './chart-model.mjs';

const $ = id => document.getElementById(id);
const h = (tag, className, value) => { const node = document.createElement(tag); if (className) node.className = className; if (value !== undefined) node.textContent = value; return node; };
const text = (id, value) => { const node = $(id), next = String(value ?? '未报告'); if (node.textContent !== next) node.textContent = next; };
const field = (id, value) => { $(id).value = Number.isFinite(value) ? String(value) : typeof value === 'string' && value ? value : '未报告'; };
const stamp = value => Number.isFinite(value) ? new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Hong_Kong', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(value) : '未记录';
const layout = initWorkspaceLayout($('workbench'));
const chartKeys = ['episodeSeconds', 'reward', 'valueLoss', 'policyLoss'];
const charts = [
  createInteractiveChart($('duration-chart'), { key: 'episodeSeconds', label: '回合时长', series: 'duration' }),
  createInteractiveChart($('reward-chart'), { key: 'reward', label: '回合奖励', series: 'reward' }),
  createInteractiveChart($('value-chart'), { key: 'valueLoss', label: 'Value Loss', series: 'loss', loss: true }),
  createInteractiveChart($('policy-chart'), { key: 'policyLoss', label: 'Policy Loss', series: 'policy', loss: true }),
];
let view = buildWorkbenchView(), snapshot = {}, busy = false, lastLearning = null, lastRun = null, lastMode = null, lastModeOptions = '';
let noticeTimer = null, selectedCheckpoint = null, parameterDraft = {}, parameterSignature = '', modelSignature = '', modelSelectMode = false;
let ioBusy = false, ioMessage = '', viewportCountBusy = false, pendingSaveId = null, supervisorKey = null, saveSubmitting = false, saveRequestError = '', reconnectBusy = false, lastActionTechnical = '';
const parameterInputs = new Map(), viewportNodes = new Map(), viewportIntents = new Map(), viewportActions = new Map();
const locallyClosedPreviews = new Set();

function disabled(id, allowed, reason, pending = busy) { const node = $(id); node.disabled = pending || !allowed; if (reason) node.title = reason; }
function notice(message) { clearTimeout(noticeTimer); text('action-notice', message); $('action-notice').hidden = false; noticeTimer = setTimeout(() => { $('action-notice').hidden = true; }, 6500); }
function friendlyActionError(error) {
  const message = error?.message || String(error), raw = typeof error?.details === 'string' ? error.details : message;
  if (/abort|timed?\s*out|timeout|failed to fetch|networkerror/i.test(raw)) { lastActionTechnical = raw; return /abort|timed?\s*out|timeout|failed to fetch|networkerror/i.test(message) ? '请求暂未确认，请稍后重试。' : message; }
  return message;
}
function renderConnection() {
  text('connection-label', view.connection.label); $('connection-status').dataset.status = view.connection.status; $('connection-status').title = view.connection.detail;
  $('reconnect-button').hidden = !['reconnecting', 'offline'].includes(view.connection.status); $('reconnect-button').disabled = reconnectBusy;
  text('connection-detail', [view.connection.detail, view.connection.technicalReason && '请求详情：' + view.connection.technicalReason, lastActionTechnical && '上次操作详情：' + lastActionTechnical].filter(Boolean).join(' '));
}
function applyProgress(node, state) {
  node.hidden = !state.active;
  if (state.percent === null) node.removeAttribute('value'); else node.value = state.percent;
  node.setAttribute('aria-label', state.label);
  node.dataset.loading = state.indeterminate ? 'true' : 'false';
}
function parameterContext() {
  const origin = $('training-origin').value;
  return origin === 'selected' ? selectedCheckpoint?.path || null : origin === 'latest' ? view.checkpoint : null;
}
function changeOrigin() {
  parameterDraft = {}; parameterSignature = '';
  snapshot = { ...snapshot, parameters: null }; view = buildWorkbenchView(snapshot);
  transport.setParameterCheckpoint(parameterContext()); render();
}
function checkedDraft() { return validateParameterDraft(view.parameterFields, parameterDraft); }
function renderActions() {
  const origin = $('training-origin').value, selected = origin === 'selected';
  const compatible = !selected || view.options.some(item => item.path === selectedCheckpoint?.path && item.compatible === true);
  const checked = checkedDraft();
  disabled('training-start', view.controls.start && compatible && checked.valid, !compatible ? '所选模型当前未通过兼容性验证。' : checked.errors.join('；') || '按所选起点与已编辑参数开始训练。');
  $('training-start').textContent = origin === 'new' ? '开始' : '继续训练';
  disabled('training-stop', view.controls.stop, '结束本轮训练并等待最终模型保存；不会只保存后台快照。');
  disabled('training-pause', view.controls.trainingPause, !view.protocol && view.active ? '当前训练未启用控制协议；下一轮训练生效，不会重启当前训练。' : '在参数回调安全边界暂停或继续；暂停不计入训练预算。');
  $('training-pause').textContent = view.phase === 'pausing' ? '正在暂停' : view.phase === 'resuming' ? '正在继续' : view.paused ? '继续' : '暂停';
  const receipt = pendingSaveId ? view.saveResults.find(item => item.requestId === pendingSaveId) : view.save;
  const pending = saveSubmitting || pendingSaveId && !receipt && !saveRequestError || receipt && ['pending', 'saving'].includes(receipt.status);
  disabled('training-save', view.controls.saveSnapshot && !pending, !view.protocol && view.active ? '当前训练未启用控制协议；下一轮训练生效，不会重启当前训练。' : '保存独立快照，不停止训练；完成以服务端回执为准。');
  $('training-save').textContent = saveSubmitting ? '提交中' : pending ? receipt?.status === 'saving' ? '保存中' : '等待保存' : '保存';
  let saveText = '';
  if (saveSubmitting) saveText = '正在提交保存请求，等待服务端确认。';
  else if (receipt?.status === 'completed') saveText = '快照已保存 · ' + checkpointLabel(receipt.checkpoint, receipt.capturedSteps, null, receipt.checkpointId);
  else if (receipt?.status === 'failed') saveText = '快照保存失败：' + (receipt.error?.message || receipt.error || '服务端未报告原因') + '；可重试。';
  else if (saveRequestError && !receipt) saveText = '保存请求未确认：' + saveRequestError + '；可重试。';
  else if (pending) saveText = receipt?.status === 'saving' ? '正在保存并核验快照，训练状态以服务端为准。' : '保存请求已接受，等待训练回调处理。';
  text('snapshot-status', saveText);
  text('save-feedback', saveText); $('save-feedback').hidden = !saveText;
  $('snapshot-status').title = receipt ? checkpointTitle(receipt.checkpoint, receipt.capturedSteps, receipt.checkpointId) : '';
  text('parameter-error', checked.errors.join('；'));
}
function renderParameters() {
  const fields = view.parameterFields;
  const signature = JSON.stringify([snapshot.legacy?.mode, view.parameterSchema?.checkpoint || null, fields.map(item => [item.key, item.label, item.type, item.min, item.max, item.step, item.group, item.editable])]);
  if (signature !== parameterSignature) {
    parameterSignature = signature; parameterInputs.clear();
    for (const id of ['parallel-parameters', 'ppo-parameters', 'eval-parameters']) $(id).replaceChildren();
    const knownIds = { num_envs: 'param-envs', seed: 'param-seed', batch_size: 'param-batch', num_minibatches: 'param-minibatches', unroll_length: 'param-rollout', num_updates_per_batch: 'param-epochs', learning_rate: 'param-lr', discounting: 'param-discount', gae_lambda: 'param-gae', clipping_epsilon: 'param-clip', num_eval_envs: 'param-eval-envs', eval_interval_seconds: 'param-eval-interval' };
    for (const item of fields) {
      const label = h('label', '', item.label || item.key), input = h('input');
      input.id = knownIds[item.key] || 'param-' + item.key.replace(/[^a-zA-Z0-9_-]/g, '-');
      input.type = item.type === 'boolean' ? 'checkbox' : 'number';
      if (Number.isFinite(item.min)) input.min = String(item.min);
      if (Number.isFinite(item.max)) input.max = String(item.max);
      input.step = Number.isFinite(item.step) ? String(item.step) : item.type === 'integer' ? '1' : 'any';
      input.addEventListener('input', () => { parameterDraft[item.key] = item.type === 'boolean' ? input.checked : input.value; renderActions(); });
      label.append(input);
      $(item.group === '并行' ? 'parallel-parameters' : item.group === '评估' ? 'eval-parameters' : 'ppo-parameters').append(label);
      parameterInputs.set(item.key, input);
    }
    for (const id of ['parallel-parameters', 'ppo-parameters', 'eval-parameters']) if (!$(id).children.length) $(id).append(h('p', 'caption', fields.length ? '未报告此类可调参数。' : '正在读取参数定义；尚不提交任何覆盖值。'));
  }
  for (const item of fields) {
    const input = parameterInputs.get(item.key), value = Object.hasOwn(parameterDraft, item.key) ? parameterDraft[item.key] : item.value;
    input.disabled = busy || !view.controls.ppoWrite || item.editable !== true;
    input.title = input.disabled ? view.active ? '训练中锁定；参数变更仅用于下一轮。' : item.editable !== true ? '此字段由训练器管理，当前不允许覆盖。' : '参数写入能力尚不可用。' : '仅在开始下一轮时提交；未编辑字段沿用自动配置或模型配置。';
    if (item.type === 'boolean') input.checked = value === true;
    else if (document.activeElement !== input) input.value = Number.isFinite(value) || typeof value === 'string' ? String(value) : '';
  }
  text('config-note', view.active ? '当前训练配置已锁定。' : Object.keys(parameterDraft).length ? '已编辑 ' + Object.keys(parameterDraft).length + ' 项；开始下一轮时提交，其余沿用自动或继承配置。' : '未编辑字段沿用自动配置或所选模型；续训不恢复优化器状态。');
}
function previewList() {
  if (view.viewports.length) return view.viewports;
  return [{ id: '1', ...(snapshot.legacy || {}), streamUrl: '/stream.mjpg', loading: !view.known ? { active: !view.error, stage: 'connecting', error: view.error } : { active: snapshot.legacy?.sync_status === 'pending', stage: snapshot.legacy?.sync_status === 'pending' ? 'syncing' : 'idle' } }];
}
function makeViewport(id) {
  const tile = h('details', 'viewport-tile'); tile.open = true; tile.dataset.viewport = id;
  const summary = h('summary'), label = h('span', 'disclosure-label', '预览 ' + id); summary.append(label); summary.setAttribute('aria-label', '折叠或展开预览 ' + id);
  const body = h('div', 'fold-body actual-viewport'), image = h('img'), empty = h('div', 'viewport-empty');
  image.alt = '机器人实际预览 ' + id; image.hidden = true;
  const title = h('p', 'viewport-empty-title'), detail = h('p', 'viewport-empty-detail'), progress = h('progress', 'inline-progress'); progress.max = 100;
  empty.append(title, detail, progress);
  const actions = h('div', 'viewport-actions'), toggle = h('button', 'quiet-button'), pause = h('button', 'quiet-button'), reset = h('button', 'quiet-button', '重置'), remove = h('button', 'quiet-button', '移除');
  for (const button of [toggle, pause, reset, remove]) button.type = 'button';
  toggle.addEventListener('click', () => { const state = previewList().find(item => String(item.id) === id); viewportAction(!view.connected ? 'suspend' : locallyClosedPreviews.has(id) || viewportNodes.get(id)?.streamError ? 'open' : state?.loading?.active || state?.suspended === false ? 'suspend' : 'open', id); });
  pause.addEventListener('click', () => { const state = previewList().find(item => String(item.id) === id); viewportAction(viewportNodes.get(id)?.streamError ? 'suspend' : state?.paused ? 'unpause' : 'pause', id); });
  reset.addEventListener('click', () => viewportAction('reset', id)); remove.addEventListener('click', () => viewportAction('remove', id));
  actions.append(toggle, pause, reset, remove); body.append(image, empty, actions); tile.append(summary, body);
  const node = { tile, image, empty, title, detail, progress, toggle, pause, reset, remove, streamKey: null, streamGeneration: null, streamIdentity: null, streamError: false, streamFailure: null };
  return node;
}
function attachViewportStream(node, id, stream, generation, identity = {}) {
  node.streamKey = stream; node.streamGeneration = generation; node.streamError = false; node.streamFailure = null;
  node.streamIdentity = stream ? { key: stream, mode: identity.mode || view.mode, runId: identity.runId ?? null, generation } : null;
  if (!stream) { node.image.removeAttribute('src'); return; }
  const image = h('img'); image.alt = '机器人实际预览 ' + id; image.hidden = true;
  const request = { key: stream, generation };
  image.addEventListener('error', () => {
    if (node.image !== image || viewportNodes.get(id) !== node) return;
    const state = previewList().find(item => String(item.id) === id);
    const failure = previewImageFailure(request, { key: node.streamKey, generation: state?.generation ?? 0, open: state?.suspended === false, connected: view.connected, intendedOpen: viewportIntents.get(id) !== false });
    if (failure) { node.streamError = true; node.streamFailure = failure; image.removeAttribute('src'); renderPreviews(); }
  });
  node.image.replaceWith(image); node.image = image; image.src = stream;
}
function renderPreviews() {
  const items = previewList(), ids = new Set(items.map(item => String(item.id)));
  for (const [id, node] of viewportNodes) if (!ids.has(id)) { node.image.removeAttribute('src'); node.tile.remove(); viewportNodes.delete(id); viewportIntents.delete(id); locallyClosedPreviews.delete(id); }
  $('viewport-grid').dataset.count = String(items.length);
  if (!viewportNodes.size) $('viewport-grid').replaceChildren();
  for (const [index, state] of items.entries()) {
    const id = String(state.id);
    let node = viewportNodes.get(id);
    if (!node) { node = makeViewport(id); viewportNodes.set(id, node); $('viewport-grid').append(node.tile); }
    const locallyClosed = locallyClosedPreviews.has(id), reportedLoading = loadingView(state.loading, '载入预览');
    const closing = !locallyClosed && viewportIntents.get(id) === false && (state.suspended === false || state.loading?.active);
    const identity = { mode: view.mode, runId: state.model_run_id || null, generation: state.generation ?? 0, known: view.known, open: state.suspended === false && (!state.mode || state.mode === view.mode), intendedOpen: !locallyClosed && viewportIntents.get(id) !== false };
    const retained = previewStreamPlan({ ...node.streamIdentity, failed: node.streamError }, identity);
    const canStream = identity.open && identity.intendedOpen && (view.connected || retained.kind !== 'close');
    let stream = null;
    if (canStream) {
      if (!view.connected) stream = retained.key;
      else try { const url = new URL(state.streamUrl || '/stream.mjpg?viewport=' + encodeURIComponent(id), location.origin); if (url.origin === location.origin) { url.searchParams.set('generation', String(state.generation ?? 0)); stream = url.pathname + url.search; } } catch {}
    }
    if (stream !== node.streamKey) attachViewportStream(node, id, stream, state.generation ?? 0, identity);
    const reported = locallyClosed ? loadingView() : closing ? loadingView({ active: true, message: '正在关闭预览' }) : node.streamFailure || reportedLoading;
    const loading = !view.connected && reported.active ? { ...reported, active: false, visible: false, indeterminate: false } : reported;
    node.tile.dataset.loading = loading.active ? 'true' : 'false';
    node.tile.dataset.error = node.streamError ? 'true' : 'false';
    node.image.id = index === 0 ? 'preview-stream' : 'preview-stream-' + id;
    node.image.hidden = !stream || loading.active || node.streamError;
    node.empty.hidden = Boolean(stream && !loading.active && !node.streamError);
    node.title.textContent = locallyClosed ? '预览已关闭（本地）' : !view.known ? '尚无预览' : loading.visible ? loading.label : !view.connected && reportedLoading.active ? reportedLoading.label + '（上次报告）' : node.streamError ? '预览流中断' : state.suspended !== false ? '预览已关闭' : state.paused ? '预览已暂停' : '预览已开启';
    node.detail.textContent = locallyClosed ? '此页已停止接收画面，需手动重新开启。' : node.streamFailure ? '重试会重新请求此窗口；也可关闭释放。' : loading.error || state.sync_error || state.error || (loading.active ? '进度由服务端报告。关闭可取消载入。' : view.known && state.suspended !== false ? '开启时同步一次，观看期间固定模型版本。' : '');
    applyProgress(node.progress, loading);
    const enabled = view.connected && (view.controls.multiWindow || index === 0 && view.controls.preview);
    const localClose = view.connection.allowLocalPreviewClose && !locallyClosed && (state.suspended === false || reportedLoading.active || Boolean(node.streamKey));
    node.toggle.disabled = !(enabled || localClose) || closing; node.toggle.textContent = locallyClosed ? '开启' : localClose ? '关闭' : closing ? '正在关闭' : loading.active ? '取消载入' : loading.error || node.streamError ? '重试' : state.suspended === false ? '关闭' : '开启';
    node.pause.disabled = !enabled || !node.streamError && (state.suspended !== false || loading.active); node.pause.textContent = node.streamError ? '关闭' : state.paused ? '继续预览' : '暂停预览';
    node.reset.disabled = !enabled || state.suspended !== false || loading.active || node.streamError;
    node.remove.hidden = items.length <= 1; node.remove.disabled = !view.controls.multiWindow;
  }
  const first = items[0] || {}, firstId = String(first.id), firstLocal = locallyClosedPreviews.has(firstId), firstFailure = viewportNodes.get(firstId)?.streamFailure, load = firstLocal ? loadingView() : firstFailure || loadingView(first.loading), opened = first.suspended === false && !firstLocal;
  const status = firstLocal ? '预览已关闭（本地）' : !view.known ? '尚无预览' : load.visible ? load.label : opened ? first.paused ? '预览已暂停' : '预览已开启' : '预览已关闭';
  text('preview-chip', status); text('viewport-status', status);
  text('preview-short-version', checkpointLabel(first.checkpoint, first.preview_training_steps));
  text('preview-description', first.model_source || '模型来源未报告');
  text('preview-version', [first.checkpoint, first.model_run_id && '运行 ' + first.model_run_id].filter(Boolean).join(' · '));
  const firstLocalClose = view.connection.allowLocalPreviewClose && !firstLocal && (opened || first.loading?.active);
  $('preview-open').textContent = firstLocalClose ? '关闭预览' : firstFailure ? '重试预览' : load.active ? '取消载入' : opened ? '关闭预览' : '开启预览';
  $('preview-pause').textContent = firstFailure ? '关闭预览' : first.paused ? '继续预览' : '暂停预览';
  disabled('preview-open', view.controls.preview || view.controls.multiWindow || firstLocalClose, firstLocalClose ? '只关闭此页图像客户端，不发送训练或服务端控制命令。' : '仅控制当前第一个预览窗口，不影响训练。', false);
  disabled('preview-pause', view.connected && opened && !load.active && (view.controls.preview || view.controls.multiWindow), firstFailure ? '关闭失败的预览并释放实例，不影响训练。' : '只暂停预览，不暂停训练。', false);
  disabled('preview-reset', view.connected && opened && !load.active && !firstFailure && (view.controls.preview || view.controls.multiWindow), '仅重置预览回合。', false);
  disabled('viewport-count', view.controls.multiWindow, '最多四个独立 CPU 预览；减少窗口会释放对应实例。', viewportCountBusy);
  if (!viewportCountBusy) $('viewport-count').value = String(items.length);
  field('param-fps', first.fps);
  const inputs = Array.isArray(first.actuator_inputs) ? first.actuator_inputs : Array.isArray(first.torques) ? first.torques : [];
  const units = Array.isArray(first.input_units) ? first.input_units : view.preview.units, order = view.preview.order;
  text('telemetry-state', !view.connected ? '最后报告' : !opened ? '最后报告 · 预览关闭' : first.paused ? '预览已暂停' : inputs.length ? '服务端报告' : '未报告');
  text('telemetry-time', '采样时间未报告' + (Number.isFinite(first.episode_seconds) ? ' · 回合 ' + first.episode_seconds.toFixed(3) + ' 秒' : ''));
  $('telemetry-values').replaceChildren(...inputs.map((value, index) => { const item = h('div'), dt = h('dt', '', order[index] || '执行器 ' + (index + 1)), dd = h('dd', '', formatChartValue(value) + (units[index] ? ' ' + units[index] : '')); item.append(dt, dd); return item; }));
  text('telemetry-description', first.actuator_description || first.action_semantics || '执行器语义未报告');
}
async function viewportAction(action, id) {
  const requestMode = previewRequestMode(action, view);
  if (requestMode === 'blocked') return;
  if (requestMode === 'local-close') { locallyClosedPreviews.add(id); viewportIntents.set(id, false); const node = viewportNodes.get(id); if (node) attachViewportStream(node, id, null, node.streamGeneration); renderPreviews(); return { localOnly: true }; }
  if (action === 'open') locallyClosedPreviews.delete(id);
  const retryNode = viewportNodes.get(id), retryKey = action === 'open' && retryNode?.streamError ? retryNode.streamKey : null, retryGeneration = retryNode?.streamGeneration;
  const refreshRetrySource = sequence => { const node = viewportNodes.get(id); if (retryKey && sequence === viewportActions.get(id) && node?.streamError && node.streamKey === retryKey && node.streamGeneration === retryGeneration) { node.streamKey = null; renderPreviews(); } };
  const sequence = (viewportActions.get(id) || 0) + 1; viewportActions.set(id, sequence);
  if (action === 'suspend' || action === 'remove') viewportIntents.set(id, false);
  if (action === 'open') viewportIntents.set(id, true);
  renderPreviews();
  try {
    if (view.controls.multiWindow) {
      const result = await transport.command({ path: '/api/workbench/v1/viewports', body: { action, id } });
      if (action === 'open' && sequence !== viewportActions.get(id) && viewportIntents.get(id) === false) await transport.command({ path: '/api/workbench/v1/viewports', body: { action: 'suspend', id } });
      refreshRetrySource(sequence);
      return result;
    }
    if (!view.controls.preview) throw new Error('当前预览操作不可用');
    const result = await transport.command({ path: '/control', body: { action: action === 'open' ? 'resume' : action } }); refreshRetrySource(sequence); return result;
  } catch (error) { notice('预览操作未完成：' + friendlyActionError(error)); }
  finally { renderPreviews(); }
}
async function resizeViewports() {
  if (viewportCountBusy || !view.controls.multiWindow) return;
  const target = Number($('viewport-count').value); viewportCountBusy = true; renderPreviews();
  try {
    let items = previewList();
    while (items.length < target) {
      const result = await transport.command({ path: '/api/workbench/v1/viewports', body: { action: 'create' } });
      if (!result.createdId) throw new Error('服务端未报告新窗口身份');
      await viewportAction('open', String(result.createdId)); items = Array.isArray(result.viewports) ? result.viewports : previewList();
    }
    while (items.length > target) { const result = await viewportAction('remove', String(items.at(-1).id)); if (!result || !Array.isArray(result.viewports) || result.viewports.length >= items.length) throw new Error('窗口移除状态尚未确认'); items = result.viewports; }
  } catch (error) { notice('窗口布局未完成：' + friendlyActionError(error)); }
  finally { viewportCountBusy = false; renderPreviews(); }
}
function renderModels() {
  text('source-alias', view.checkpointLabel); $('source-alias').title = view.checkpointTitle;
  text('training-model-name', view.runId || '未报告'); text('training-model-steps', formatNumber(view.progress.segmentSteps) + ' 步');
  text('saved-model-name', view.checkpointLabel); $('saved-model-name').title = view.checkpointTitle;
  text('saved-model-steps', view.training.can_resume === true ? '监督器已核验可恢复' : '可恢复状态未确认');
  const first = previewList()[0] || {};
  text('view-model-name', checkpointLabel(first.checkpoint, first.preview_training_steps)); $('view-model-name').title = checkpointTitle(first.checkpoint, first.preview_training_steps);
  text('view-model-steps', first.model_run_id || '运行身份未报告');
  text('checkpoint-dir-current', view.storage?.saveRoot || '保存目录未报告');
  disabled('inspect-checkpoints', view.connected, '查看服务端返回的历史模型与兼容结果。', false);
  disabled('choose-model', view.controls.historyResume, view.active ? '训练中不能更换下一轮起点。' : '只可选择已经通过兼容性验证的精确路径。');
  disabled('checkpoint-dir-open', view.controls.directoryOpen, '由本机桥接验证后打开保存目录。', ioBusy);
  disabled('checkpoint-dir-choose', view.controls.directory, view.active ? '当前训练保存位置已锁定，结束后可更改。' : '打开原生目录选择器，取消不会更改保存位置。', ioBusy);
  disabled('checkpoint-import', view.controls.modelImport, '完整复制并验证模型后，才加入可选历史模型。', ioBusy);
  text('directory-status', ioMessage); applyProgress($('directory-progress'), loadingView({ active: ioBusy, message: ioMessage }));
  const options = view.options.length ? view.options : view.candidates.map(item => ({ ...item, path: item.relativePath, steps: item.capturedSteps, compatible: false, reason: '兼容性尚未核验。' }));
  const signature = JSON.stringify([options, modelSelectMode, view.controls.historyResume, selectedCheckpoint?.path]);
  if (signature === modelSignature) return; modelSignature = signature;
  const rows = options.map(item => {
    const row = h('div', 'model-option'), title = h('strong', '', checkpointLabel(item.path, item.steps, null, item.id)), path = h('span', '', item.path || '路径未报告');
    title.title = checkpointTitle(item.path, item.steps, item.id);
    const note = h('small', '', '运行 ' + (item.runId || '未报告') + ' · ' + (item.compatible === true ? '兼容验证通过 · 优化器重新建立' : item.reason || '兼容性未确认'));
    const button = h('button', 'quiet-button', selectedCheckpoint?.path === item.path ? '已选为下轮起点' : '从此模型续训');
    button.type = 'button'; button.disabled = !modelSelectMode || !view.controls.historyResume || item.compatible !== true;
    button.title = item.compatible === true ? '使用此精确路径；不会替换为最新模型。' : item.reason || '此模型当前不兼容。';
    button.addEventListener('click', () => { selectedCheckpoint = { ...item }; $('selected-checkpoint-option').hidden = false; $('selected-checkpoint-option').textContent = checkpointLabel(item.path, item.steps, null, item.id); $('training-origin').value = 'selected'; changeOrigin(); $('model-dialog').close(); notice('已选作下一轮起点；网络与归一化恢复，优化器重新建立。'); });
    row.append(title, path, note, button); return row;
  });
  if (!rows.length) rows.push(h('p', 'caption', '未报告历史模型，或仍在读取兼容性结果。'));
  $('model-options').replaceChildren(...rows);
}
async function directoryAction(action) {
  if (ioBusy) return; ioBusy = true; const operationMode = view.mode;
  ioMessage = action === 'open' ? '正在打开保存目录' : '等待系统目录选择；可在原生对话框中取消。'; renderModels();
  try {
    if (action === 'open') {
      if (!view.controls.directoryOpen || !view.storage?.saveRoot) throw new Error('保存目录或本机桥接尚不可用');
      await transport.command({ path: '/api/local/directories', body: { action: 'open', path: view.storage.saveRoot }, timeoutMs: 30000 });
      ioMessage = '已向系统提交打开目录请求。';
    } else {
      if (!(action === 'save' ? view.controls.directory : view.controls.modelImport)) throw new Error('当前状态不支持此目录操作');
      const result = await transport.command({ path: '/api/local/directories', body: { action: 'select', purpose: action, ...(view.storage?.saveRoot ? { path: view.storage.saveRoot } : {}) }, timeoutMs: 310000 });
      if (result.cancelled || !result.path) { ioMessage = '已取消目录选择，保存位置未改变。'; return; }
      if (action === 'import' && operationMode !== view.mode) throw new Error('目录选择期间模式已变化，请按当前模式重新导入');
      ioMessage = action === 'save' ? '正在验证保存目录' : '正在完整复制并验证导入模型'; renderModels();
      if (action === 'save') { await transport.command({ path: '/api/workbench/v1/storage', body: { path: result.path } }); ioMessage = '后续训练保存目录已更新。'; }
      else { const imported = await transport.command({ path: '/api/workbench/v1/import', body: { path: result.path, mode: operationMode }, timeoutMs: 130000 }); ioMessage = imported.checkpoint?.compatible === true ? '模型导入并验证完成，可在历史模型中选择。' : '导入回执未确认兼容性，请检查历史模型结果。'; }
    }
  } catch (error) { ioMessage = '目录操作未完成：' + friendlyActionError(error); }
  finally { ioBusy = false; renderModels(); }
}
function render() {
  document.body.dataset.connection = view.connection.status; document.body.dataset.stale = view.known && !view.connected ? 'true' : 'false';
  document.body.dataset.phase = view.active && !view.progress.preparing && ['training', 'running'].includes(view.phase) ? 'training' : view.phase;
  document.body.dataset.loading = view.activity.active ? 'true' : 'false';
  text('run-name', view.runId || (view.known ? view.modeLabel : '正在连接')); $('run-name').title = view.training.run_dir || view.mode || '';
  text('run-mode', view.modeLabel); text('phase-chip', view.trainingPhaseLabel); text('nav-checkpoint', view.known ? view.checkpointLabel : '—'); $('nav-checkpoint').title = view.checkpointTitle;
  text('settings-state', view.known ? '服务端配置' : '未报告'); text('training-active-label', view.trainingPhaseLabel);
  const optionKey = JSON.stringify(view.modes.map(item => [item.id, item.label]));
  if (optionKey !== lastModeOptions) { const options = view.modes.map(item => { const option = h('option', '', item.label); option.value = item.id; return option; }); if (!options.length && !view.mode) options.push(h('option', '', '等待服务端')); $('param-mode').replaceChildren(...options); lastModeOptions = optionKey; }
  if (view.mode) { if (![...$('param-mode').options].some(option => option.value === view.mode)) { const option = h('option', '', view.mode); option.value = view.mode; $('param-mode').append(option); } $('param-mode').value = view.mode; }
  disabled('param-mode', view.controls.mode, view.active ? '训练进行中不能切换模式。' : '切换服务端动作接口。');
  disabled('duration-select', view.controls.budget, '下一轮新增预算，0 为不限时。');
  disabled('training-origin', view.controls.start, '选择下一轮新训练、最近模型或已验证历史模型。');
  $('resume-latest-option').disabled = !view.controls.resumeLatest;
  if (!view.controls.resumeLatest && $('training-origin').value === 'latest' && !view.active && view.known) $('training-origin').value = 'new';
  const origin = $('training-origin').value;
  text('training-origin-note', !view.known ? '等待已核验的模型状态。' : origin === 'selected' ? '精确起点：' + (selectedCheckpoint?.path || '尚未选择') + '；不恢复优化器状态。' : origin === 'latest' ? '最近已核验：' + view.checkpointLabel + '；网络与归一化恢复，优化器重新建立。' : '从新策略开始；仅编辑过的字段覆盖自动配置。');
  text('action-hint', view.active ? view.paused ? '暂停时间不计入预算；可继续、保存快照或停止并保存。' : '停止会结束本轮并保存；底部“保存”只创建独立快照。' : '预算、模型与参数在开始下一轮时一起提交。');
  text('runtime-summary', Number.isFinite(view.config.num_envs) ? '当前报告 ' + formatNumber(view.config.num_envs) + ' 个并行环境。' : '当前运行配置未报告。');
  field('param-hz', view.controlHz === null ? null : view.controlHz + ' Hz'); field('param-obs', view.obsDim); text('setting-frequency', view.controlHz === null ? '未报告' : view.controlHz + ' Hz');
  const profile = view.config.runtime; text('runtime-profile', typeof profile?.effective === 'string' ? profile.effective : profile ? '已报告' : '未报告'); $('runtime-profile').title = profile ? JSON.stringify(profile) : '当前运行未报告 runtime_profile。';
  const descriptor = view.descriptor;
  text('robot-id', descriptor?.robot?.id || '未记录'); text('robot-model', descriptor?.robot?.modelId || '未记录'); text('task-id', descriptor?.task?.id || '未记录'); text('task-profile', descriptor?.task?.profile || '未记录');
  text('observation-version', descriptor?.observation?.version || '未记录'); text('observation-dims', descriptor?.observation ? 'Actor ' + (descriptor.observation.actorDim ?? '未知') + ' / Critic ' + (descriptor.observation.criticDim ?? '未知') : '未记录');
  text('action-mode', descriptor?.action?.mode || '未记录'); text('action-order', Array.isArray(descriptor?.action?.order) ? descriptor.action.order.join(' · ') : '未记录');
  text('physics-frequency', Number.isFinite(descriptor?.timing?.physicsHz) ? descriptor.timing.physicsHz + ' Hz' : '未记录'); text('episode-limit', Number.isFinite(descriptor?.timing?.episodeSeconds) ? descriptor.timing.episodeSeconds + ' 秒' : '未记录');
  text('descriptor-note', descriptor ? '已保存的运行描述 · ' + descriptor.runId : '当前运行缺少匹配的保存描述；未知字段保持空缺。');
  text('process-description', view.trainingPhaseLabel); text('reported-stage', view.training.stage || view.training.status?.stage || '未报告'); text('preparation-time', formatSeconds(view.progress.preparationSeconds));
  const started = view.training.training_clock_started ?? view.training.status?.training_clock_started;
  text('training-clock-state', view.paused ? '暂停中，计时已冻结' : started === true ? '已开始' : started === false ? '准备中，尚未计入训练' : '未报告');
  text('preparation-message', view.training.preparation_message || view.training.status?.preparation_message || ''); text('progress-note', view.progress.detail); text('training-error', view.training.error || view.training.status?.error || view.controlResult.status === 'failed' && view.controlResult.error || '');
  text('transport-summary', view.trainingPhaseLabel); text('transport-status', view.known && view.activity.visible ? view.activity.label : view.trainingPhaseLabel); text('clock-label', view.progress.label); text('run-clock', view.progress.clock);
  const percent = view.activity.active ? view.activity.percent : view.progress.percent;
  text('transport-percent', !view.known ? '—' : percent === null ? view.activity.active ? '进行中' : '未知' : Math.max(0, Math.min(100, percent)).toFixed(1) + '%');
  text('transport-detail', view.activity.active ? view.activity.label + (view.activity.percent === null ? '，进度尚无已知分母。' : '') : view.progress.detail);
  text('steps-label', view.progress.trainedSteps === null ? '本轮环境步' : '累计环境步'); text('run-steps', formatNumber(view.progress.trainedSteps ?? view.progress.segmentSteps));
  if (percent === null) $('training-progress').removeAttribute('value'); else $('training-progress').value = Math.max(0, Math.min(100, percent));
  $('training-progress').hidden = !view.known || Boolean(view.activity.error);
  $('training-progress').dataset.loading = view.activity.indeterminate ? 'true' : 'false';
  $('training-progress-track').style.setProperty('--progress-fill', (percent === null ? 0 : Math.max(0, Math.min(100, percent))) + '%');
  const currentRun = runKey(snapshot.legacy), series = chartKeys.map(key => metricRecords(view.curve.points, key));
  if (snapshot.learning !== lastLearning || currentRun !== lastRun) { charts.forEach((chart, index) => chart.setData(series[index])); lastLearning = snapshot.learning; lastRun = currentRun; }
  const latest = series.map(records => records.at(-1));
  text('metric-duration', formatChartValue(latest[0]?.episodeSeconds)); text('metric-reward', formatChartValue(latest[1]?.reward)); text('value-latest', formatChartValue(latest[2]?.valueLoss)); text('policy-latest', formatChartValue(latest[3]?.policyLoss));
  const evaluations = new Set([...series[0], ...series[1]]), lastEvaluation = view.curve.points.findLast(point => evaluations.has(point));
  const evalCount = view.curve.matches && Number.isFinite(view.curve.metadata.eval_point_count) && view.curve.metadata.eval_point_count >= 0 ? view.curve.metadata.eval_point_count : null;
  text('metric-points', evalCount === null ? '评估点数未报告' : formatNumber(evalCount) + ' 个评估点'); text('metric-samples', view.curve.metadata.eval_envs ?? '未报告'); text('chart-eval-step', formatNumber(lastEvaluation?.steps)); text('evaluation-time', stamp(lastEvaluation?.at));
  text('evaluation-context', view.curve.points.length ? '实际记录' : '暂无记录'); text('learning-source', view.curve.identity ? '运行 ' + view.curve.identity : '运行身份未报告');
  text('learning-note', view.curve.reason || (Array.isArray(view.curve.metadata.limitations) ? view.curve.metadata.limitations.join('；') : '悬停查看原始记录；缺失数据保留断点。'));
  text('source-notes', '活动状态、暂停与保存回执来自监督器；累计和运行描述按模式与运行身份校验。');
  text('read-time', view.connection.lastSuccessAt ? '页面最近成功读取 ' + stamp(view.connection.lastSuccessAt) + '；评估与遥测时间由数据源提供。' : '页面读取时间未知。'); text('warning-list', view.warnings.join('；'));
  renderConnection(); renderParameters(); renderActions(); renderPreviews(); renderModels();
}
const transport = createWorkbenchTransport({ onSnapshot(next) {
  snapshot = next; view = buildWorkbenchView(next);
  const nextKey = runKey(next.legacy); if (supervisorKey && nextKey !== supervisorKey) { pendingSaveId = null; saveRequestError = ''; } supervisorKey = nextKey;
  if (view.mode && lastMode && view.mode !== lastMode) { selectedCheckpoint = null; parameterDraft = {}; parameterSignature = ''; $('selected-checkpoint-option').hidden = true; $('training-origin').value = 'new'; transport.setParameterCheckpoint(null); }
  if (view.mode) lastMode = view.mode;
  render();
} });
async function act(action, options = {}) {
  if (busy) return;
  let request;
  try {
    if (action === 'start') {
      const origin = $('training-origin').value, checked = checkedDraft();
      if (!checked.valid) throw new Error(checked.errors.join('；'));
      options = { durationSeconds: Number($('duration-select').value), resume: origin === 'latest', checkpoint: origin === 'selected' ? selectedCheckpoint?.path : null, parameters: checked.values };
    }
    const retryId = action === 'saveSnapshot' && saveRequestError && pendingSaveId && !view.saveResults.some(item => item.requestId === pendingSaveId) ? pendingSaveId : null;
    request = commandRequest(action, view, { ...options, requestId: retryId || globalThis.crypto?.randomUUID?.() || null });
  } catch (error) { notice(error.message); return; }
  busy = true; if (action === 'saveSnapshot') { pendingSaveId = request.body.request_id || null; saveSubmitting = true; saveRequestError = ''; } renderActions();
  try {
    const result = await transport.command(request);
    if (action === 'saveSnapshot') { pendingSaveId = result.requestId || null; notice('保存请求已接受，完成情况以服务端回执为准。'); }
    else if (action === 'trainingPause') notice('控制请求已接受，等待训练安全边界确认。');
    else { if (action === 'start') parameterDraft = {}; notice('请求已提交，实际状态以服务端报告为准。'); }
  } catch (error) { const message = friendlyActionError(error); if (action === 'saveSnapshot') saveRequestError = message; notice('操作未完成：' + message); transport.refresh(); }
  finally { busy = false; saveSubmitting = false; render(); }
}
$('training-start').addEventListener('click', () => act('start')); $('training-stop').addEventListener('click', () => act('stop'));
$('training-pause').addEventListener('click', () => act('trainingPause')); $('training-save').addEventListener('click', () => act('saveSnapshot'));
$('training-origin').addEventListener('change', changeOrigin); $('param-mode').addEventListener('change', event => act('mode', { mode: event.target.value }));
$('preview-open').addEventListener('click', () => { const item = previewList()[0]; viewportAction(!view.connected ? 'suspend' : locallyClosedPreviews.has(String(item.id)) || viewportNodes.get(String(item.id))?.streamError ? 'open' : item?.loading?.active || item?.suspended === false ? 'suspend' : 'open', String(item.id)); });
$('preview-pause').addEventListener('click', () => { const item = previewList()[0]; viewportAction(viewportNodes.get(String(item.id))?.streamError ? 'suspend' : item.paused ? 'unpause' : 'pause', String(item.id)); });
$('preview-reset').addEventListener('click', () => viewportAction('reset', String(previewList()[0].id)));
$('viewport-count').addEventListener('change', resizeViewports);
$('layout-reset').addEventListener('click', () => layout.reset()); $('reconnect-button').addEventListener('click', async event => { event.preventDefault(); event.stopPropagation(); if (reconnectBusy) return; reconnectBusy = true; renderConnection(); try { await transport.refresh(); } catch (error) { lastActionTechnical = typeof error?.details === 'string' ? error.details : error?.message || String(error); } finally { reconnectBusy = false; renderConnection(); } });
$('inspect-checkpoints').addEventListener('click', () => { modelSelectMode = false; renderModels(); $('model-dialog').showModal(); });
$('choose-model').addEventListener('click', () => { modelSelectMode = true; renderModels(); $('model-dialog').showModal(); });
$('close-model-dialog').addEventListener('click', () => $('model-dialog').close());
$('checkpoint-dir-open').addEventListener('click', () => directoryAction('open'));
$('checkpoint-dir-choose').addEventListener('click', () => directoryAction('save'));
$('checkpoint-import').addEventListener('click', () => directoryAction('import'));
window.addEventListener('pagehide', () => { transport.stop(); charts.forEach(chart => chart.destroy()); layout.destroy(); for (const node of viewportNodes.values()) node.image.removeAttribute('src'); }, { once: true });
render(); transport.start();
