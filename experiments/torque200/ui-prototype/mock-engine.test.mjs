import test from 'node:test';
import assert from 'node:assert/strict';
import { SCENARIOS, createDemo, transition, tick } from './mock-engine.mjs';

test('fixtures are independent, synthetic and have consistent evaluation identity', () => {
  for (const { id } of SCENARIOS) {
    const state = createDemo(id);
    assert.equal(state.demo, true);
    assert.ok(state.run.id.startsWith('demo-'));
    assert.ok(state.evaluation.points.every(point => point.at <= state.now));
    if (state.evaluation.points.length) assert.equal(state.data.evaluationAt, state.evaluation.points.at(-1).at);
    if (state.preview.sampleAt !== null) assert.ok(state.preview.sampleAt >= state.preview.syncedAt);
  }
  const first = createDemo();
  first.models[0].name = 'changed';
  assert.notEqual(createDemo().models[0].name, 'changed');
  const state = createDemo();
  assert.equal(state.data.evaluationSteps, state.evaluation.points.at(-1).steps);
  assert.ok(state.data.evaluationSteps < state.run.steps);
});

test('active runs reject duplicate start and source changes without mutation', () => {
  for (const scenario of ['training', 'paused', 'preparing', 'saving', 'verifying']) {
    const original = createDemo(scenario);
    const before = structuredClone(original);
    const started = transition(original, 'start', { modelId: null });
    assert.equal(started.run.id, original.run.id);
    assert.equal(started.phase, original.phase);
    assert.ok(started.notice);
    const selected = transition(original, 'select-model', { modelId: null });
    assert.equal(selected.selectedModelId, original.selectedModelId);
    assert.deepEqual(original, before);
  }
});

test('model choice requires compatible verified source and only edits next-run config', () => {
  const state = createDemo('ready');
  const incompatible = transition(state, 'select-model', { modelId: 'demo-torque-policy' });
  assert.equal(incompatible.selectedModelId, state.selectedModelId);
  assert.ok(incompatible.notice);
  const selected = transition(state, 'select-model', { modelId: 'demo-model-019' });
  assert.equal(selected.selectedModelId, 'demo-model-019');
  assert.equal(selected.phase, 'idle');
  const started = transition(selected, 'start', { durationSeconds: 0 });
  assert.equal(started.run.parentModelId, 'demo-model-019');
  assert.equal(started.run.schedule, 'unlimited');
  assert.equal(started.run.steps, 0);
});

test('new strategy can start with no saved model; invalid budgets do not start', () => {
  const empty = createDemo('empty');
  assert.equal(transition(empty, 'start', { modelId: null, durationSeconds: 900 }).phase, 'preparing');
  for (const value of [-1, NaN, Infinity, 1, 604801]) {
    assert.equal(transition(empty, 'start', { durationSeconds: value }).phase, 'idle');
  }
});

test('cancel and failed-stage retry preserve completed preparation evidence', () => {
  const preparing = createDemo('preparing');
  const cancelled = transition(preparing, 'stop');
  assert.equal(cancelled.phase, 'cancelled');
  assert.equal(cancelled.preparation.stages[0].status, 'cached');
  assert.equal(cancelled.preparation.stages[1].status, 'cancelled');
  const failed = createDemo('prepare-failed');
  const restarted = transition(failed, 'start', { modelId: null });
  assert.equal(restarted.preparation.stages[0].status, 'cached');
  assert.equal(restarted.preparation.stages[1].status, 'waiting');
  const progressed = tick(restarted, 14);
  assert.ok(['compiling', 'training'].includes(progressed.phase));
  assert.ok(progressed.preparation.stages.every(stage => ['cached', 'passed'].includes(stage.status)));
});

test('stopping is cooperative and creates no verified result before verification completes', () => {
  let state = transition(createDemo(), 'stop');
  const oldModel = state.latestSavedModelId;
  assert.equal(state.phase, 'stopping');
  for (const phase of ['saving', 'exporting', 'verifying']) {
    state = tick(state, 2);
    assert.equal(state.phase, phase);
    assert.equal(state.latestSavedModelId, oldModel);
  }
  state = tick(state, 2);
  assert.equal(state.phase, 'completed');
  assert.notEqual(state.latestSavedModelId, oldModel);
  assert.equal(state.models.find(model => model.id === state.latestSavedModelId).verified, true);
  assert.equal(createDemo('save-failed').latestSavedModelId, oldModel);
});

test('preview opens once, pins one policy and does not control the training lifecycle', () => {
  let state = transition(createDemo(), 'preview-open');
  assert.equal(state.preview.status, 'syncing');
  state = tick(state, 2);
  const version = state.preview.steps;
  assert.equal(state.preview.status, 'running');
  state = tick(state, 20);
  assert.ok(state.run.steps > version);
  assert.equal(state.preview.steps, version);
  state = transition(state, 'preview-open');
  assert.equal(state.preview.steps, version);
  state = transition(state, 'preview-toggle-pause');
  assert.equal(state.preview.status, 'paused');
  assert.equal(state.phase, 'training');
  const sample = state.preview.sampleAt;
  state = tick(state, 2);
  assert.equal(state.preview.sampleAt, sample);
  state = transition(state, 'preview-close');
  assert.equal(state.phase, 'training');
  assert.equal(state.preview.status, 'off');
  assert.equal(state.preview.sampleAt, sample);
});

test('closed preview cannot reset or fabricate a live sample', () => {
  const state = createDemo();
  const reset = transition(state, 'preview-reset');
  assert.equal(reset.preview.status, 'off');
  assert.equal(reset.preview.sampleAt, null);
  assert.equal(reset.preview.telemetry.episodeSeconds, null);
});

test('idle preview identifies the saved source and preparation without a model cannot preview', () => {
  let state = createDemo('ready');
  const saved = state.models.find(model => model.id === state.latestSavedModelId);
  state = tick(transition(state, 'preview-open'), 2);
  assert.equal(state.preview.modelId, saved.id);
  assert.equal(state.preview.steps, saved.steps);
  assert.equal(transition(createDemo('preparing'), 'preview-open').preview.status, 'off');
});

test('starting a new run clears the previous preview identity and samples', () => {
  const previous = tick(transition(createDemo('ready'), 'preview-open'), 2);
  assert.ok(previous.preview.modelId);
  assert.ok(previous.preview.sampleAt);
  const started = transition(previous, 'start', { durationSeconds: 900 });
  assert.equal(started.phase, 'preparing');
  assert.equal(started.preview.status, 'off');
  for (const key of ['modelId', 'runId', 'steps', 'syncedAt', 'sampleAt', 'error']) {
    assert.equal(started.preview[key], null);
  }
  assert.ok(Object.values(started.preview.telemetry).every(value => value === null));
  assert.equal(started._clock.preview, 0);
  assert.ok(previous.preview.modelId);
});

test('disconnect keeps run identity and phase; reconnect does not create a run', () => {
  const state = createDemo('disconnected');
  const later = tick(state, 60);
  assert.equal(later.run.steps, state.run.steps);
  assert.equal(later.phase, 'training');
  assert.equal(later.data.receivedAt, state.data.receivedAt);
  assert.equal(transition(later, 'stop').phase, 'training');
  const reconnected = transition(later, 'reconnect');
  assert.equal(reconnected.connection, 'connected');
  assert.equal(reconnected.run.id, state.run.id);
  assert.equal(reconnected.selectedModelId, state.selectedModelId);
  assert.equal(reconnected.data.receivedAt, reconnected.now);
});

test('unknown-length compile and timed boundary have separate progression', () => {
  let state = createDemo('compiling');
  state.run.durationSeconds = 6;
  state = tick(state, 4);
  assert.equal(state.phase, 'training');
  assert.equal(state.run.steps, 0);
  state = tick(state, 2);
  assert.equal(state.phase, 'stopping');
  assert.equal(state.run.elapsedSeconds, 6);
});

test('training and evaluation refresh separately; missing losses stay missing', () => {
  let state = createDemo();
  const evalAt = state.data.evaluationAt;
  state = tick(state, 10);
  assert.equal(state.data.evaluationAt, evalAt);
  assert.ok(state.run.steps > state.data.evaluationSteps);
  state = tick(state, 17);
  assert.equal(state.data.evaluationAt, state.now);
  assert.equal(state.data.evaluationSteps, state.run.steps);
  assert.equal(state.evaluation.points[0].valueLoss, null);
  const stale = tick(createDemo('stale'), 20);
  assert.equal(stale.data.progressAt, createDemo('stale').data.progressAt);
  assert.equal(stale.data.receivedAt, stale.now);
});

test('training pause freezes training progress while an open preview remains independent', () => {
  const running = createDemo('preview-running');
  const paused = transition(running, 'pause-training');
  assert.equal(paused.phase, 'paused');
  const later = tick(paused, 45);
  assert.equal(later.run.steps, paused.run.steps);
  assert.equal(later.run.elapsedSeconds, paused.run.elapsedSeconds);
  assert.deepEqual(later.evaluation, paused.evaluation);
  assert.equal(later.data.progressAt, paused.data.progressAt);
  assert.equal(later.data.receivedAt, later.now);
  assert.equal(later.now, paused.now + 45000);
  assert.equal(later.preview.status, 'running');
  assert.equal(later.preview.sampleAt, later.now);
  assert.equal(later.preview.steps, running.preview.steps);
  const resumed = tick(transition(later, 'resume-training'), 1);
  assert.equal(resumed.run.id, running.run.id);
  assert.equal(resumed.run.steps, running.run.steps + 8192);
  assert.equal(resumed.run.elapsedSeconds, running.run.elapsedSeconds + 1);
});

test('resume does not create duplicate runs, and paused training stops through normal verification', () => {
  const paused = createDemo('paused');
  const resumed = transition(paused, 'resume-training');
  const repeated = transition(resumed, 'resume-training');
  assert.equal(repeated.phase, 'training');
  assert.equal(repeated.run.id, paused.run.id);
  assert.equal(repeated.run.steps, paused.run.steps);
  assert.ok(repeated.notice);
  const stopped = tick(transition(paused, 'stop'), 8);
  assert.equal(stopped.phase, 'completed');
  assert.equal(stopped.run.steps, paused.run.steps);
});

test('snapshot fixes request steps and publishes only after verification without stopping training', () => {
  const before = createDemo();
  let saving = transition(before, 'save-snapshot');
  const requestSteps = saving.run.steps;
  const requestAt = saving.now;
  const previousModel = saving.latestSavedModelId;
  assert.equal(saving.checkpointSave.phase, 'saving');
  assert.equal(saving.checkpointSave.steps, requestSteps);
  assert.equal(saving.checkpointSave.requestedAt, requestAt);
  assert.equal(transition(saving, 'save-snapshot').checkpointSave.requestedAt, requestAt);
  saving = tick(saving, 2);
  assert.equal(saving.phase, 'training');
  assert.equal(saving.checkpointSave.phase, 'verifying');
  assert.equal(saving.latestSavedModelId, previousModel);
  assert.equal(saving.checkpointSave.modelId, null);
  assert.ok(saving.run.steps > requestSteps);
  const saved = tick(saving, 2);
  assert.equal(saved.phase, 'training');
  assert.equal(saved.checkpointSave.phase, 'saved');
  const model = saved.models.find(value => value.id === saved.checkpointSave.modelId);
  assert.equal(model.steps, requestSteps);
  assert.equal(model.verified, true);
  assert.equal(saved.latestSavedModelId, model.id);
  assert.equal(saved.run.elapsedSeconds, before.run.elapsedSeconds + 4);
  assert.equal(before.checkpointSave.phase, 'idle');
});

test('snapshot and lifecycle save are exclusive; paused snapshot completes without advancing training', () => {
  const paused = createDemo('paused');
  const saving = transition(paused, 'save-snapshot');
  assert.equal(transition(saving, 'stop').phase, 'paused');
  const saved = tick(saving, 4);
  assert.equal(saved.phase, 'paused');
  assert.equal(saved.run.steps, paused.run.steps);
  assert.equal(saved.run.elapsedSeconds, paused.run.elapsedSeconds);
  assert.equal(saved.checkpointSave.phase, 'saved');
  const stopping = transition(saved, 'stop');
  const denied = transition(stopping, 'save-snapshot');
  assert.equal(denied.phase, 'stopping');
  assert.equal(denied.checkpointSave.phase, 'saved');
  assert.ok(denied.notice);
  const completed = tick(stopping, 8);
  assert.equal(completed.phase, 'completed');
  assert.equal(completed.models.find(value => value.id === completed.latestSavedModelId).steps, paused.run.steps);
});

test('snapshot pauses its save clock while disconnected and resumes after reconnect', () => {
  let saving = transition(createDemo(), 'save-snapshot');
  saving.connection = 'disconnected';
  const waiting = tick(saving, 30);
  assert.equal(waiting.checkpointSave.phase, 'saving');
  assert.equal(waiting._clock.checkpointSave, 0);
  assert.equal(waiting.run.steps, saving.run.steps);
  assert.equal(transition(createDemo('disconnected'), 'save-snapshot').checkpointSave.phase, 'idle');
  const saved = tick(transition(waiting, 'reconnect'), 4);
  assert.equal(saved.checkpointSave.phase, 'saved');
  assert.equal(saved.checkpointSave.steps, saving.checkpointSave.steps);
});

test('timed stopping waits for the in-flight snapshot to finish before lifecycle save starts', () => {
  let state = createDemo();
  state.run.durationSeconds = state.run.elapsedSeconds + 1;
  state = transition(state, 'save-snapshot');
  state = tick(state, 3);
  assert.equal(state.phase, 'training');
  assert.equal(state.checkpointSave.phase, 'verifying');
  state = tick(state, 1);
  assert.equal(state.phase, 'training');
  assert.equal(state.checkpointSave.phase, 'saved');
  state = tick(state, 1);
  assert.equal(state.phase, 'stopping');
});

test('continuing a model loads cumulative history while this-run counters start at zero', () => {
  const selected = transition(createDemo('ready'), 'select-model', { modelId: 'demo-model-019' });
  const started = transition(selected, 'start', { durationSeconds: 600 });
  assert.equal(started.run.parentModelId, 'demo-model-019');
  assert.equal(started.run.baseTrainedSeconds, 360);
  assert.equal(started.run.baseTrainedSteps, 327680);
  assert.equal(started.run.elapsedSeconds, 0);
  assert.equal(started.run.steps, 0);
  assert.equal(started.models.find(model => model.id === 'demo-model-019').steps, 327680);
});

test('snapshot and final metadata accumulate parent history without changing existing steps meaning', () => {
  const original = createDemo();
  const saved = tick(transition(original, 'save-snapshot'), 4);
  const checkpoint = saved.models.find(model => model.id === saved.latestSavedModelId);
  assert.equal(checkpoint.steps, original.run.steps);
  assert.equal(checkpoint.trainedSteps, original.run.baseTrainedSteps + original.run.steps);
  assert.equal(checkpoint.trainedSeconds, original.run.baseTrainedSeconds + original.run.elapsedSeconds);
  assert.ok(checkpoint.trainedSeconds < saved.run.baseTrainedSeconds + saved.run.elapsedSeconds);
  const completed = tick(transition(saved, 'stop'), 8);
  const finalModel = completed.models.find(model => model.id === completed.latestSavedModelId);
  assert.equal(finalModel.steps, saved.run.steps);
  assert.equal(finalModel.trainedSteps, saved.run.baseTrainedSteps + saved.run.steps);
  assert.equal(finalModel.trainedSeconds, saved.run.baseTrainedSeconds + saved.run.elapsedSeconds);
});

test('legacy and author history remain unknown; an explicitly new strategy has known zero history', () => {
  const legacy = createDemo('ready');
  const oldModel = legacy.models.find(model => model.id === legacy.selectedModelId);
  delete oldModel.trainedSeconds;
  delete oldModel.trainedSteps;
  let started = transition(legacy, 'start', { durationSeconds: 600 });
  assert.equal(started.run.baseTrainedSeconds, null);
  assert.equal(started.run.baseTrainedSteps, null);
  started = tick(started, 7);
  const saved = tick(transition(started, 'save-snapshot'), 4);
  const checkpoint = saved.models.find(model => model.id === saved.latestSavedModelId);
  assert.equal(checkpoint.trainedSeconds, null);
  assert.equal(checkpoint.trainedSteps, null);
  const completed = tick(transition(saved, 'stop'), 8);
  const finalModel = completed.models.find(model => model.id === completed.latestSavedModelId);
  assert.equal(finalModel.trainedSeconds, null);
  assert.equal(finalModel.trainedSteps, null);
  const author = transition(createDemo('ready'), 'start', { modelId: 'demo-author-policy', durationSeconds: 600 });
  assert.equal(author.run.baseTrainedSeconds, null);
  assert.equal(author.run.baseTrainedSteps, null);
  const fresh = transition(createDemo('empty'), 'start', { modelId: null, durationSeconds: 600 });
  assert.equal(fresh.run.baseTrainedSeconds, 0);
  assert.equal(fresh.run.baseTrainedSteps, 0);
});

test('resume-progress begins halfway through the combined budget and advances the same run', () => {
  const state = createDemo('resume-progress');
  assert.equal(state.phase, 'paused');
  assert.equal(state.run.parentModelId, 'demo-model-023');
  assert.equal(state.run.baseTrainedSeconds, 600);
  assert.equal(state.run.baseTrainedSteps, 524288);
  assert.equal(state.run.elapsedSeconds, 0);
  assert.equal(state.run.steps, 0);
  assert.equal(state.run.durationSeconds, 600);
  assert.equal(state.run.baseTrainedSeconds / (state.run.baseTrainedSeconds + state.run.durationSeconds), 0.5);
  const resumed = tick(transition(state, 'resume-training'), 1);
  assert.equal(resumed.run.id, state.run.id);
  assert.equal(resumed.run.elapsedSeconds, 1);
  assert.equal(resumed.run.steps, 8192);
  assert.equal(resumed.run.baseTrainedSeconds + resumed.run.elapsedSeconds, 601);
  assert.equal(resumed.run.baseTrainedSteps + resumed.run.steps, 532480);
});
