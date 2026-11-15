import test from 'node:test';
import assert from 'node:assert/strict';
import { fitPaneSizes, resizePair, createPaneGesture, updatePaneGesture, PANE_MINIMUMS, COLLAPSED_WIDTH } from './layout.mjs';

const total = sizes => sizes.reduce((value, size) => value + size, 0);
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-7, `${actual} != ${expected}`);

test('default wide layout leaves the center the remaining width', () => {
  assert.deepEqual(fitPaneSizes(1440), [280, 844, 300]);
  near(total(fitPaneSizes(1024)), 1008);
  fitPaneSizes(1024).forEach((width, index) => assert.ok(width >= PANE_MINIMUMS[index]));
});

test('container changes preserve preferred proportions subject to pane minima', () => {
  const preferred = [360, 760, 304];
  const wide = fitPaneSizes(1600, preferred);
  near(wide[0] / wide[1], preferred[0] / preferred[1]);
  const constrained = fitPaneSizes(780, [10, 1000, 10]);
  near(constrained[0], 200);
  near(constrained[2], 230);
  near(total(constrained), 764);
  assert.deepEqual(preferred, [360, 760, 304]);
});

test('each collapsed pane keeps a compact rail and reopening restores its share', () => {
  const preferred = [280, 844, 300];
  for (let index = 0; index < 3; index += 1) {
    const open = [true, true, true];
    open[index] = false;
    const folded = fitPaneSizes(1440, preferred, open);
    assert.equal(folded[index], COLLAPSED_WIDTH);
    near(total(folded), 1424);
    assert.deepEqual(fitPaneSizes(1440, preferred), preferred);
  }
  assert.deepEqual(fitPaneSizes(1440, preferred, [false, false, false]), [48, 48, 48]);
});

test('a single open pane fills the remaining space without losing closed rails', () => {
  assert.deepEqual(fitPaneSizes(1440, [280, 844, 300], [false, true, false]), [48, 1328, 48]);
});

test('very small or invalid containers return finite nonnegative widths within budget', () => {
  for (const width of [0, 12, 100, 340, 600, NaN, Infinity, -20]) {
    for (const open of [[true, true, true], [false, true, false], [false, false, false]]) {
      const fitted = fitPaneSizes(width, [NaN, -20, Infinity], open);
      assert.ok(fitted.every(value => Number.isFinite(value) && value >= 0));
      const budget = Number.isFinite(width) ? Math.max(0, width - 16) : 0;
      assert.ok(total(fitted) <= budget + 1e-7);
    }
  }
});

test('left separator exchanges only left and center widths and clamps both boundaries', () => {
  const source = [280, 844, 300];
  assert.deepEqual(resizePair(source, 'left', 60), [340, 784, 300]);
  assert.deepEqual(resizePair(source, 'left', -10000), [200, 924, 300]);
  assert.deepEqual(resizePair(source, 'left', 10000), [864, 260, 300]);
  assert.deepEqual(source, [280, 844, 300]);
});

test('right separator adjusts the center and right panes without moving the left pane', () => {
  assert.deepEqual(resizePair([280, 844, 300], 'right', 40), [280, 884, 260]);
  assert.deepEqual(resizePair([280, 844, 300], 1, 10000), [280, 914, 230]);
  assert.deepEqual(resizePair([280, 844, 300], 1, -10000), [280, 260, 884]);
});

test('pair resizing degrades safely when the pair cannot meet desktop minima', () => {
  const next = resizePair([60, 120, 150], 0, 10000);
  near(total(next), 330);
  assert.ok(next.every(Number.isFinite));
  near(next[0] / next[1], 200 / 260);
});

test('raw width thresholds have a hysteresis band instead of flickering at the minimum', () => {
  let gesture = createPaneGesture([280, 844, 300], [true, true, true], 'left');
  gesture = updatePaneGesture(gesture, -119);
  assert.equal(gesture.open[0], true);
  assert.equal(gesture.sizes[0], 200);
  gesture = updatePaneGesture(gesture, -120);
  assert.equal(gesture.open[0], false);
  assert.equal(gesture.sizes[0], 48);
  gesture = updatePaneGesture(gesture, -57);
  assert.equal(gesture.open[0], false);
  gesture = updatePaneGesture(gesture, -56);
  assert.equal(gesture.open[0], true);
  assert.equal(gesture.sizes[0], 224);
  near(total(gesture.sizes), 1424);
  assert.equal(gesture.sizes[2], 300);
});

test('right and center panes can collapse from either adjacent boundary without moving the unrelated pane', () => {
  const initial = [280, 844, 300];
  const right = updatePaneGesture(createPaneGesture(initial, [true, true, true], 'right'), 110);
  assert.deepEqual(right.open, [true, true, false]);
  assert.deepEqual(right.sizes, [280, 1096, 48]);
  const centerFromLeft = updatePaneGesture(createPaneGesture(initial, [true, true, true], 'left'), 624);
  assert.deepEqual(centerFromLeft.open, [true, false, true]);
  assert.deepEqual(centerFromLeft.sizes, [1076, 48, 300]);
  const centerFromRight = updatePaneGesture(createPaneGesture(initial, [true, true, true], 'right'), -624);
  assert.deepEqual(centerFromRight.open, [true, false, true]);
  assert.deepEqual(centerFromRight.sizes, [280, 48, 1096]);
});

test('a closed neighboring rail restores through a continuous drag and keeps total width', () => {
  const left = updatePaneGesture(createPaneGesture([48, 1076, 300], [false, true, true], 'left'), 176);
  assert.deepEqual(left.open, [true, true, true]);
  assert.deepEqual(left.sizes, [224, 900, 300]);
  const right = updatePaneGesture(createPaneGesture([280, 1096, 48], [true, true, false], 'right'), -206);
  assert.deepEqual(right.open, [true, true, true]);
  assert.deepEqual(right.sizes, [280, 890, 254]);
  const center = updatePaneGesture(createPaneGesture([1076, 48, 300], [true, false, true], 'right'), 236);
  assert.deepEqual(center.open, [true, true, false]);
  assert.deepEqual(center.sizes, [1076, 300, 48]);
  for (const gesture of [left, right, center]) near(total(gesture.sizes), 1424);
});

test('two closed adjacent rails can recover by borrowing bounded space from the remaining pane', () => {
  const gesture = updatePaneGesture(createPaneGesture([48, 48, 1328], [false, false, true], 'left'), 176);
  assert.deepEqual(gesture.open, [true, false, true]);
  assert.deepEqual(gesture.sizes, [224, 48, 1152]);
  near(total(gesture.sizes), 1424);
  assert.ok(gesture.sizes[2] >= PANE_MINIMUMS[2]);
});

test('last-open protection, extreme deltas and untouched inputs remain safe', () => {
  const initial = [48, 260, 48];
  const open = [false, true, false];
  const protectedGesture = updatePaneGesture(createPaneGesture(initial, open, 'left'), 100);
  assert.deepEqual(protectedGesture.open, open);
  assert.deepEqual(protectedGesture.sizes, initial);
  for (const boundary of ['left', 'right']) {
    for (const delta of [-Infinity, -1e12, 0, 1e12, Infinity, NaN]) {
      const gesture = updatePaneGesture(createPaneGesture([280, 844, 300], [true, true, true], boundary), delta);
      assert.ok(gesture.open.some(Boolean));
      assert.ok(gesture.sizes.every(value => Number.isFinite(value) && value >= 0));
      near(total(gesture.sizes), 1424);
    }
  }
  assert.deepEqual(initial, [48, 260, 48]);
  assert.deepEqual(open, [false, true, false]);
});

test('container fitting alone never changes open flags or triggers gesture collapse', () => {
  const open = [true, true, true];
  const fitted = fitPaneSizes(400, [280, 844, 300], open);
  const gesture = updatePaneGesture(createPaneGesture(fitted, open, 'left'), 0);
  assert.deepEqual(gesture.open, open);
});
