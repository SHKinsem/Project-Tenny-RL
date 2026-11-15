import test from 'node:test';
import assert from 'node:assert/strict';
import { createChartModel, createCurveFixture, formatChartValue } from './chart-model.mjs';

const points = values => values.map((value, steps) => ({ steps, at: steps, value }));
const flattened = model => model.segments.flat();

test('dense pixel sampling bounds display count while retaining source extrema and endpoints', () => {
  for (const kind of ['dense-1k', 'dense-10k']) {
    const fixture = createCurveFixture(kind);
    const model = createChartModel(fixture, 'valueLoss', { width: 160 });
    assert.equal(model.records.length, fixture.length);
    assert.ok(model.sampledCount <= 4 * (160 - 16));
    assert.ok(model.sampledCount < model.validCount);
    const sampled = new Set(flattened(model).map(record => record.index));
    const highest = model.records.reduce((a, b) => a.value > b.value ? a : b);
    const lowest = model.records.reduce((a, b) => a.value < b.value ? a : b);
    assert.ok(sampled.has(highest.index));
    assert.ok(sampled.has(lowest.index));
    assert.ok(sampled.has(0));
    assert.ok(sampled.has(fixture.length - 1));
  }
});

test('hover selects an exact unsampled original record, independently of line sampling', () => {
  const fixture = createCurveFixture('dense-10k');
  const model = createChartModel(fixture, 'reward', { width: 120 });
  const displayed = new Set(flattened(model).map(record => record.index));
  const original = model.records.find(record => !displayed.has(record.index));
  const hovered = model.nearestAtRatio(original.point.steps / fixture.at(-1).steps);
  assert.equal(hovered, original);
  assert.equal(hovered.point, fixture[original.index]);
  assert.equal(model.recordAtIndex(original.index), original);
  assert.equal(model.recordAtIndex(-1), null);
  assert.equal(model.recordAtIndex(1.5), null);
});

test('missing runs break line segments and remain selectable without being filled with zero', () => {
  const fixture = createCurveFixture('gaps');
  const model = createChartModel(fixture, 'valueLoss', { width: 180 });
  assert.equal(model.totalCount, 1000);
  assert.equal(model.validCount, 907);
  assert.equal(model.segments.length, 4);
  assert.ok(model.sampledCount <= 4 * (180 - 16 + model.segments.length));
  for (const segment of model.segments) {
    assert.ok(segment.every(record => record.value !== null && Number.isFinite(record.y)));
    for (let index = 1; index < segment.length; index += 1) {
      const between = model.records.slice(segment[index - 1].index, segment[index].index + 1);
      assert.ok(between.every(record => record.value !== null));
    }
  }
  const missing = model.records[225];
  assert.equal(model.nearestAtRatio(missing.point.steps / fixture.at(-1).steps), missing);
  assert.equal(missing.value, null);
  assert.equal(missing.y, null);
  assert.ok(Number.isFinite(missing.x));
});

test('wide positive losses select log and retain numeric order without changing raw values', () => {
  const fixture = createCurveFixture('wide-loss');
  const model = createChartModel(fixture, 'valueLoss', { loss: true });
  assert.equal(model.scale, 'log');
  assert.equal(model.domain[0], 1e-10);
  assert.equal(model.domain[1], 1e3);
  assert.ok(model.records[0].y < model.records.at(-1).y);
  assert.equal(model.records.at(-1).value, fixture.at(-1).valueLoss);
  assert.equal(createChartModel(points([1, 2, 3]), 'value', { loss: true }).scale, 'linear');
});

test('signed losses use symlog, and explicit log safely falls back for zero or negative data', () => {
  const model = createChartModel(createCurveFixture('signed-loss'), 'policyLoss', { loss: true });
  assert.equal(model.scale, 'symlog');
  assert.ok(model.constant > 0 && Number.isFinite(model.constant));
  assert.ok(model.domain[0] < 0 && model.domain[1] > 0);
  const positive = model.records.find(record => record.value > 0);
  const zero = model.records.find(record => record.value === 0);
  const negative = model.records.find(record => record.value < 0);
  assert.ok(positive.y < zero.y && zero.y < negative.y);
  const fallback = createChartModel(points([0, 1e-7, -1e-8, 2e-6]), 'value', { scale: 'log' });
  assert.equal(fallback.scale, 'symlog');
  assert.match(fallback.reason, /回退/);
  assert.deepEqual(fallback.records.map(record => record.value), [0, 1e-7, -1e-8, 2e-6]);
  assert.equal(createChartModel(points([0, 1, 2]), 'value', { loss: true }).scale, 'linear');
});

test('tiny variations use the real span rather than being flattened by a fixed padding', () => {
  const model = createChartModel(createCurveFixture('tiny-changes'), 'valueLoss', { loss: true });
  assert.equal(model.scale, 'linear');
  assert.ok(model.domain[1] - model.domain[0] < 1e-10);
  const ys = model.records.map(record => record.y);
  assert.ok(Math.max(...ys) - Math.min(...ys) > model.height * .6);
  assert.match(formatChartValue(model.domain[0]), /e-/);
  assert.ok(Number(formatChartValue(model.domain[0])) > 0);
  assert.equal(formatChartValue(1e-9), '1.00000e-9');
  assert.notEqual(formatChartValue(1e-12), '0');
  assert.equal(formatChartValue(-0), '0');
  assert.equal(formatChartValue(NaN), '—');
});

test('constants, zeros, missing data and extreme finite values never generate NaN coordinates', () => {
  for (const values of [[0, 0, 0], [1e-9, 1e-9, 1e-9], [-Number.MAX_VALUE, 0, Number.MAX_VALUE], [Number.MIN_VALUE, 1e-300]]) {
    const model = createChartModel(points(values), 'value', { loss: true });
    assert.ok(model.records.every(record => Number.isFinite(record.x) && Number.isFinite(record.y)));
    assert.ok(model.constant > 0 && Number.isFinite(model.constant));
    if (values.every(value => value === values[0])) assert.ok(model.records.every(record => record.y === model.height / 2));
  }
  const missing = createChartModel(points([null, NaN, Infinity, undefined]), 'value');
  assert.equal(missing.validCount, 0);
  assert.deepEqual(missing.domain, [null, null]);
  assert.deepEqual(missing.segments, []);
  assert.ok(missing.records.every(record => Number.isFinite(record.x) && record.y === null));
  assert.equal(createChartModel([], 'value').nearestAtRatio(.5), null);
});

test('sorting is stable for duplicate steps and fixtures are deterministic independent arrays', () => {
  const source = [{ steps: 2, value: 8 }, { steps: 1, value: null }, { steps: 1, value: 3 }];
  const before = structuredClone(source);
  const model = createChartModel(source, 'value');
  assert.deepEqual(model.records.map(record => record.point), [source[1], source[2], source[0]]);
  assert.equal(model.nearestAtRatio(0).point, source[1]);
  assert.deepEqual(source, before);
  const duplicate = createChartModel([{ steps: 5, value: 1 }, { steps: 5, value: 2 }], 'value');
  assert.ok(duplicate.records.every(record => record.x === duplicate.width / 2));
  const now = Date.parse('2026-11-15T10:00:00+08:00');
  for (const kind of ['dense-1k', 'dense-10k', 'wide-loss', 'signed-loss', 'tiny-changes', 'gaps', 'constant']) {
    const first = createCurveFixture(kind, { endStep: 12345, now });
    const second = createCurveFixture(kind, { endStep: 12345, now });
    assert.deepEqual(first, second);
    assert.equal(first.at(-1).steps, 12345);
    assert.ok(first.every(point => point.at <= now));
    first[0].reward = -123;
    assert.notEqual(second[0].reward, -123);
  }
});
