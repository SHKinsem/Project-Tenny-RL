// Offline executable reference. The product page does not import this file.
import assert from 'node:assert/strict';
import { createChartModel, createCurveFixture, formatChartValue } from '../chart-model.mjs';
import { createPaneGesture, updatePaneGesture, PANE_MINIMUMS, COLLAPSE_OFFSET, EXPAND_OFFSET, COLLAPSED_WIDTH } from '../layout.mjs';
import { SCENARIOS, createDemo, transition, tick } from '../mock-engine.mjs';

const NOW = Date.parse('2026-11-15T12:45:00+08:00');
const END_STEP = 573440;
const fixture = kind => createCurveFixture(kind, { endStep: END_STEP, now: NOW });
const total = sizes => sizes.reduce((value, size) => value + size, 0);

const dense = fixture('dense-10k');
const sampling = [160, 280, 640].map(width => {
  const model = createChartModel(dense, 'valueLoss', { width, loss: true });
  const visible = new Set(model.segments.flat().map(record => record.index));
  const original = model.records.find(record => !visible.has(record.index));
  assert.ok(original, 'The dense fixture should contain points omitted from the line sample.');
  const hovered = model.nearestAtRatio(original.point.steps / END_STEP);
  assert.equal(hovered.point, dense[original.index]);
  assert.ok(model.sampledCount < model.totalCount);
  return { width, rawPoints: model.totalCount, linePoints: model.sampledCount,
    unsampledHover: { index: hovered.index, step: hovered.point.steps, raw: hovered.value,
      displayed: formatChartValue(hovered.value) } };
});

const wide = createChartModel(fixture('wide-loss'), 'valueLoss', { loss: true });
const signedPoints = fixture('signed-loss');
const signed = createChartModel(signedPoints, 'policyLoss', { loss: true });
const logFallback = createChartModel(signedPoints, 'policyLoss', { scale: 'log', loss: true });
const tiny = createChartModel(fixture('tiny-changes'), 'valueLoss', { loss: true });
assert.equal(wide.scale, 'log');
assert.equal(signed.scale, 'symlog');
assert.equal(logFallback.scale, 'symlog');
assert.ok(signed.domain[0] < 0 && signed.domain[1] > 0 && signed.records.some(record => record.value === 0));
assert.ok(tiny.domain[0] > 0 && tiny.domain[1] - tiny.domain[0] < 1e-10);
const scales = {
  wideLoss: { scale: wide.scale, rawRange: wide.domain, smallestValue: formatChartValue(wide.domain[0]) },
  signedLoss: { scale: signed.scale, rawRange: signed.domain, zeroRemains: 0,
    requestedLogFallsBackTo: logFallback.scale, constant: signed.constant },
  tinyChanges: { scale: tiny.scale, rawRange: tiny.domain,
    displayedRange: tiny.domain.map(formatChartValue) },
};

let gesture = createPaneGesture([280, 844, 300], [true, true, true], 'left');
const gestureRows = [];
let collapsed;
for (const [label, delta] of [['before-collapse', -119], ['collapse', -120], ['reverse-inside-band', -57], ['expand', -56]]) {
  gesture = updatePaneGesture(gesture, delta);
  assert.equal(total(gesture.sizes), 1424);
  assert.equal(gesture.sizes[2], 300);
  if (label === 'collapse') collapsed = gesture;
  gestureRows.push({ label, delta, rawLeft: gesture.rawSizes[0], open: gesture.open, sizes: gesture.sizes });
}
assert.deepEqual(gestureRows.map(row => row.open[0]), [true, false, false, true]);
const restored = updatePaneGesture(createPaneGesture(collapsed.sizes, collapsed.open, 'left'), 176);
assert.equal(restored.open[0], true);
assert.equal(restored.sizes[0], PANE_MINIMUMS[0] + EXPAND_OFFSET);
assert.equal(total(restored.sizes), 1424);
const paneGesture = {
  collapseRawLeftAt: PANE_MINIMUMS[0] - COLLAPSE_OFFSET,
  expandRawLeftAt: PANE_MINIMUMS[0] + EXPAND_OFFSET,
  collapsedRail: COLLAPSED_WIDTH, paneWidthTotal: 1424,
  sequence: gestureRows,
  freshDragFromRail: { delta: 176, open: restored.open, sizes: restored.sizes },
};

// This is a display projection of existing state fields, not a second engine.
function progressView(state) {
  const { baseTrainedSeconds, elapsedSeconds, durationSeconds } = state.run;
  const knownHistory = Number.isFinite(baseTrainedSeconds) && baseTrainedSeconds >= 0;
  const cumulativeSeconds = knownHistory ? baseTrainedSeconds + elapsedSeconds : null;
  const cumulativeTargetSeconds = knownHistory && durationSeconds > 0 ? baseTrainedSeconds + durationSeconds : null;
  return { phase: state.phase, historySeconds: baseTrainedSeconds, addedSeconds: elapsedSeconds,
    addedBudgetSeconds: durationSeconds, cumulativeSeconds, cumulativeTargetSeconds,
    cumulativePercent: cumulativeTargetSeconds > 0 ? Number((100 * cumulativeSeconds / cumulativeTargetSeconds).toFixed(6)) : null,
    segmentPercent: durationSeconds > 0 ? Number((100 * elapsedSeconds / durationSeconds).toFixed(6)) : null };
}

assert.ok(SCENARIOS.some(scenario => scenario.id === 'resume-progress'));
const initial = createDemo('resume-progress');
const running = tick(transition(initial, 'resume-training'), 60);
const paused = tick(transition(running, 'pause-training'), 120);
const resumed = tick(transition(paused, 'resume-training'), 60);
assert.equal(progressView(initial).cumulativePercent, 50);
assert.equal(progressView(initial).segmentPercent, 0);
assert.equal(progressView(running).cumulativePercent, 55);
assert.equal(progressView(running).segmentPercent, 10);
assert.equal(paused.run.elapsedSeconds, running.run.elapsedSeconds);
assert.equal(paused.run.steps, running.run.steps);
assert.equal(progressView(resumed).cumulativePercent, 60);

const source = createDemo('ready');
const author = source.models.find(model => model.id === 'demo-author-policy');
assert.ok(author && author.trainedSeconds === null && author.trainedSteps === null);
const selected = transition(source, 'select-model', { modelId: author.id });
const unknown = tick(transition(selected, 'start', { durationSeconds: 600 }), 20);
assert.equal(unknown.run.baseTrainedSeconds, null);
assert.equal(unknown.run.baseTrainedSteps, null);
assert.equal(progressView(unknown).cumulativePercent, null);
assert.equal(progressView(unknown).cumulativeTargetSeconds, null);
assert.ok(progressView(unknown).segmentPercent > 0);
const resumeProgress = {
  initial: progressView(initial), after60Seconds: progressView(running),
  after120PausedSeconds: progressView(paused), afterResume60Seconds: progressView(resumed),
  unknownHistory: { ...progressView(unknown), historySteps: unknown.run.baseTrainedSteps },
};

console.log(JSON.stringify({ sampling, scales, paneGesture, resumeProgress }, null, 2));
