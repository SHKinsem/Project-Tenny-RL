import test from 'node:test';
import assert from 'node:assert/strict';
import { buildWorkbenchView, connectionView, previewImageFailure, previewRequestMode, previewStreamPlan } from './adapter.mjs';

const legacy = { mode: 'fixture-mode', suspended: false, training: { active: false, phase: 'completed', run_id: 'run-a', elapsed_seconds: 900, duration_seconds: 900, steps: 11862016, can_resume: false } };
const capabilities = { legacyControls: { startTraining: true, preview: true, switchMode: true, multipleViewports: true } };
const source = { legacy, capabilities, catalog: { modes: [{ id: 'fixture-mode', label: '已保存模式' }] }, connected: true };
test('单次超时只显示重连，保留真实训练阶段与计数，技术细节不混入主文案', () => {
  const view = buildWorkbenchView({ ...source, connected: false, connection: { status: 'reconnecting', failures: 1, lastSuccessAt: 2000000000000, reason: 'timeout', details: 'AbortError: signal is aborted without reason' }, error: '请求超时' });
  assert.equal(view.connection.label, '正在重连'); assert.equal(view.phaseLabel, '已完成'); assert.equal(view.trainingPhaseLabel, '已完成');
  assert.equal(view.progress.segmentSteps, 11862016); assert.equal(view.progress.elapsedSeconds, 900); assert.equal(view.connected, false);
  assert.ok(Object.values(view.controls).every(value => value === false)); assert.ok(!/Abort|signal/.test(view.connection.label + view.connection.detail)); assert.match(view.connection.technicalReason, /AbortError/);
});
test('持续失败显示已断开，恢复清除原因并重新开放经能力确认的操作', () => {
  const offline = buildWorkbenchView({ ...source, connected: false, connection: { status: 'offline', failures: 5, lastSuccessAt: 2000000000000, reason: 'server' } });
  assert.equal(offline.connection.label, '连接已断开'); assert.equal(offline.phaseLabel, '已完成'); assert.equal(offline.controls.start, false);
  const recovered = buildWorkbenchView({ ...source, connection: { status: 'connected', failures: 0, lastSuccessAt: 2000000001000, reason: null, details: null } });
  assert.equal(recovered.connection.label, '已连接'); assert.equal(recovered.connected, true); assert.equal(recovered.controls.start, true); assert.equal(recovered.connection.technicalReason, null);
  assert.equal(recovered.progress.segmentSteps, offline.progress.segmentSteps);
});
test('状态标签不能覆盖新鲜度门禁，旧snapshot也提供稳定的连接信息', () => {
  const contradictory = buildWorkbenchView({ ...source, connected: false, connection: { status: 'connected', failures: 0 } });
  assert.equal(contradictory.connected, false); assert.equal(contradictory.controls.start, false); assert.equal(contradictory.connection.status, 'reconnecting');
  const previous = buildWorkbenchView({ ...source, connected: false, error: 'AbortError' });
  assert.equal(previous.connection.status, 'offline'); assert.equal(previous.trainingPhaseLabel, '已完成');
  assert.equal(connectionView(null).label, '正在连接');
});
test('陈旧状态只允许本地关闭预览，不允许打开/重置/暂停或任何训练写入', () => {
  const view = buildWorkbenchView({ ...source, connected: false, connection: { status: 'reconnecting' } });
  assert.equal(previewRequestMode('suspend', view), 'local-close');
  for (const action of ['open', 'pause', 'unpause', 'reset', 'create', 'remove']) assert.equal(previewRequestMode(action, view), 'blocked');
  assert.equal(previewRequestMode('suspend', buildWorkbenchView(source)), 'remote');
  assert.equal(previewRequestMode('suspend', buildWorkbenchView()), 'blocked');
});
test('单次状态失败仅保留既有同版本健康流，不新建或跨mode/run/gen保留', () => {
  const request = { key: '/stream.mjpg?viewport=1&generation=7', mode: 'mode-a', runId: 'saved-run', generation: 7, failed: false };
  const current = { mode: 'mode-a', runId: 'saved-run', generation: 7, known: true, open: true, intendedOpen: true };
  assert.deepEqual(previewStreamPlan(request, current), { kind: 'retain', key: request.key });
  for (const change of [{ mode: 'mode-b' }, { runId: 'another-run' }, { generation: 8 }, { open: false }, { intendedOpen: false }, { known: false }]) assert.equal(previewStreamPlan(request, { ...current, ...change }).kind, 'close');
  assert.equal(previewStreamPlan(null, current).kind, 'close');
  assert.equal(previewStreamPlan({ ...request, failed: true }, current).kind, 'failed');
});
test('状态重连时真实图像error仍退出等待，而关闭/旧代次error仍忽略', () => {
  const request = { key: '/stream.mjpg?generation=7', generation: 7 };
  const current = { ...request, open: true, connected: false, intendedOpen: true };
  assert.equal(previewImageFailure(request, current).active, false); assert.equal(previewImageFailure(request, current).stage, 'error');
  assert.equal(previewImageFailure(request, { ...current, intendedOpen: false }), null); assert.equal(previewImageFailure(request, { ...current, generation: 8 }), null);
});
