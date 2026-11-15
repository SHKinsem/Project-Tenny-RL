// Standalone pane layout. No storage, transport, or training dependencies.
export const PANE_MINIMUMS = Object.freeze([200, 260, 230]);
export const COLLAPSED_WIDTH = 48;
export const SPLITTER_WIDTH = 8;
export const COLLAPSE_OFFSET = 40;
export const EXPAND_OFFSET = 24;
const names = ['left', 'center', 'right'];
const labels = ['曲线', '预览', '参数'];
const positive = (value, fallback = 0) => Number.isFinite(value) ? Math.max(0, value) : fallback;
const sum = values => values.reduce((total, value) => total + value, 0);
const defaults = width => [280, Math.max(260, positive(width) - 2 * SPLITTER_WIDTH - 580), 300];

/** Fit pane weights into the container, retaining a 48px rail for closed panes. */
export function fitPaneSizes(containerWidth, preferredSizes = defaults(containerWidth), openStates = [true, true, true]) {
  const budget = Math.max(0, positive(containerWidth) - 2 * SPLITTER_WIDTH);
  const open = names.map((_, index) => openStates[index] !== false);
  const preferred = names.map((_, index) => positive(preferredSizes[index], PANE_MINIMUMS[index]) || PANE_MINIMUMS[index]);
  const result = open.map(value => value ? 0 : COLLAPSED_WIDTH);
  const fixed = sum(result);
  if (fixed > budget) return result.map(value => fixed ? value / fixed * budget : 0);
  const available = budget - fixed;
  const indices = names.map((_, index) => index).filter(index => open[index]);
  if (!indices.length) return result;
  const minimumTotal = sum(indices.map(index => PANE_MINIMUMS[index]));
  if (available < minimumTotal) {
    for (const index of indices) result[index] = available * PANE_MINIMUMS[index] / minimumTotal;
    return result;
  }

  let remaining = available;
  let flexible = [...indices];
  while (flexible.length) {
    const totalWeight = sum(flexible.map(index => preferred[index]));
    const constrained = flexible.filter(index => remaining * preferred[index] / totalWeight < PANE_MINIMUMS[index]);
    if (!constrained.length) {
      for (const index of flexible) result[index] = remaining * preferred[index] / totalWeight;
      break;
    }
    for (const index of constrained) { result[index] = PANE_MINIMUMS[index]; remaining -= result[index]; }
    flexible = flexible.filter(index => !constrained.includes(index));
  }
  // Assign the floating-point remainder to one open pane, never to a closed rail.
  const last = indices.at(-1);
  result[last] = Math.max(0, budget - sum(result.filter((_, index) => index !== last)));
  return result;
}

function bounds(sizes, index, minimums = PANE_MINIMUMS) {
  const total = positive(sizes[index]) + positive(sizes[index + 1]);
  const first = positive(minimums[index]);
  const second = positive(minimums[index + 1]);
  const scale = total < first + second && first + second > 0 ? total / (first + second) : 1;
  return [first * scale, Math.max(0, total - second * scale)];
}

/** Move one separator; the two adjacent panes exchange width, others stay fixed. */
export function resizePair(sizes, boundary, delta, minimums = PANE_MINIMUMS) {
  const index = boundary === 'right' || boundary === 1 ? 1 : 0;
  const next = names.map((_, pane) => positive(sizes[pane]));
  const total = next[index] + next[index + 1];
  const [minimum, maximum] = bounds(next, index, minimums);
  next[index] = Math.min(maximum, Math.max(minimum, next[index] + (Number.isFinite(delta) ? delta : 0)));
  next[index + 1] = Math.max(0, total - next[index]);
  return next;
}

/** Snapshot one pointer/keyboard gesture; delta is always relative to its start. */
export function createPaneGesture(sizes, openStates, boundary) {
  const initial = names.map((_, index) => positive(sizes[index]));
  const open = names.map((_, index) => openStates[index] !== false);
  return { index: boundary === 'right' || boundary === 1 ? 1 : 0,
    startSizes: [...initial], startOpen: [...open], sizes: [...initial], open, rawSizes: [...initial], delta: 0 };
}

/** Apply hysteresis to raw widths, then conserve the neighboring pair's width. */
export function updatePaneGesture(gesture, requestedDelta) {
  const index = gesture.index;
  const other = index === 0 ? 2 : 0;
  const pair = [index, index + 1];
  const initial = gesture.startSizes;
  const total = sum(initial);
  const limit = total + sum(PANE_MINIMUMS) + 100;
  const delta = typeof requestedDelta === 'number' && !Number.isNaN(requestedDelta)
    ? Math.max(-limit, Math.min(limit, requestedDelta)) : 0;
  const raw = [...initial];
  raw[index] += delta;
  raw[index + 1] -= delta;
  const open = [...gesture.open];
  for (const pane of pair) {
    if (open[pane] && (!gesture.startOpen[pane] || raw[pane] < initial[pane]) && raw[pane] <= PANE_MINIMUMS[pane] - COLLAPSE_OFFSET) open[pane] = false;
    else if (!open[pane] && raw[pane] >= PANE_MINIMUMS[pane] + EXPAND_OFFSET) open[pane] = true;
  }
  const initiallyBothClosed = !gesture.startOpen[index] && !gesture.startOpen[index + 1];
  const pairTotal = initial[index] + initial[index + 1];
  if (!initiallyBothClosed && open[index] && open[index + 1] && pairTotal < PANE_MINIMUMS[index] + PANE_MINIMUMS[index + 1]) {
    // Do not squeeze two expanded panes below their minima. The new pane can
    // open once the opposite one crosses its collapse threshold instead.
    for (const pane of pair) if (!gesture.open[pane]) open[pane] = false;
  }
  if (!initiallyBothClosed && !open[index] && !open[index + 1]) {
    open[gesture.open[index] ? index : index + 1] = true;
  }
  if (gesture.open.some(Boolean) && !open.some(Boolean)) {
    open[gesture.open.findIndex(Boolean)] = true;
  }

  const sizes = [...initial];
  let allocation = pairTotal;
  if (initiallyBothClosed && (open[index] || open[index + 1])) {
    const expanded = open[index] ? index : index + 1;
    const outsideMinimum = open[other] ? PANE_MINIMUMS[other] : COLLAPSED_WIDTH;
    const capacity = Math.max(0, total - outsideMinimum);
    const required = Math.max(PANE_MINIMUMS[expanded], raw[expanded]) + COLLAPSED_WIDTH;
    if (capacity < PANE_MINIMUMS[expanded] + COLLAPSED_WIDTH) open[expanded] = false;
    else allocation = Math.min(capacity, Math.max(pairTotal, required));
  }
  if (open[index] && open[index + 1]) {
    const resized = resizePair(initial, index, delta);
    sizes[index] = resized[index]; sizes[index + 1] = resized[index + 1];
  } else if (open[index]) {
    sizes[index + 1] = Math.min(COLLAPSED_WIDTH, allocation);
    sizes[index] = Math.max(0, allocation - sizes[index + 1]);
  } else if (open[index + 1]) {
    sizes[index] = Math.min(COLLAPSED_WIDTH, allocation);
    sizes[index + 1] = Math.max(0, allocation - sizes[index]);
  } else {
    sizes[index] = Math.min(COLLAPSED_WIDTH, allocation / 2);
    sizes[index + 1] = Math.min(COLLAPSED_WIDTH, allocation - sizes[index]);
  }
  if (initiallyBothClosed) sizes[other] = Math.max(0, total - sizes[index] - sizes[index + 1]);
  return { ...gesture, sizes, open, rawSizes: raw, delta };
}

/** Bind the five-child details / separator / details / separator / details grid. */
export function initWorkspaceLayout(root = globalThis.document?.getElementById('workbench')) {
  if (!root) throw new Error('Workspace layout requires a workbench element.');
  const children = [...root.children];
  const panes = names.map(name => children.find(node => node.getAttribute('data-pane') === name));
  const handles = ['left', 'right'].map(name => children.find(node => node.getAttribute('data-splitter') === name));
  if (panes.some(node => !node || node.tagName.toLowerCase() !== 'details') || handles.some(node => !node)) {
    throw new Error('Workspace layout requires three direct details panes and two separators.');
  }
  const doc = root.ownerDocument;
  const view = doc.defaultView;
  const media = view.matchMedia('(max-width: 760px)');
  const measure = () => Math.max(0, root.clientWidth);
  const openStates = () => panes.map(pane => pane.open);
  let preferred = defaults(measure());
  let sizes = [0, 0, 0];
  let drag = null;
  let keyboard = null;
  let destroyed = false;
  let lastKey = '';
  const ownToggles = new Map();

  function enabled() { return !media.matches; }
  function aria() {
    handles.forEach((handle, index) => {
      const canMove = enabled(index);
      const bothClosed = !panes[index].open && !panes[index + 1].open;
      const outside = index === 0 ? 2 : 0;
      const maximum = bothClosed ? sum(sizes) - PANE_MINIMUMS[outside] - COLLAPSED_WIDTH
        : sizes[index] + sizes[index + 1] - COLLAPSED_WIDTH;
      const number = value => String(Number(positive(value).toFixed(1)));
      handle.setAttribute('role', 'separator');
      handle.setAttribute('aria-orientation', 'vertical');
      handle.setAttribute('aria-label', `调整${labels[index]}与${labels[index + 1]}宽度`);
      handle.setAttribute('aria-disabled', String(!canMove));
      handle.setAttribute('aria-valuemin', number(COLLAPSED_WIDTH));
      handle.setAttribute('aria-valuemax', number(Math.max(COLLAPSED_WIDTH, maximum)));
      handle.setAttribute('aria-valuenow', number(sizes[index]));
      handle.setAttribute('aria-valuetext', `${labels[index]}${panes[index].open ? '' : '已收起'} ${Math.round(sizes[index])} 像素，${labels[index + 1]}${panes[index + 1].open ? '' : '已收起'} ${Math.round(sizes[index + 1])} 像素`);
      handle.setAttribute('aria-description', '拖窄自动收起，拖回展开。方向键调宽，Home收起前栏，End收起后栏。');
      handle.setAttribute('aria-keyshortcuts', 'ArrowLeft ArrowRight Home End');
      handle.setAttribute('aria-controls', `${panes[index].id} ${panes[index + 1].id}`);
      handle.tabIndex = canMove ? 0 : -1;
    });
  }
  function paint() {
    names.forEach((name, index) => root.style.setProperty(`--${name}-size`, `${positive(sizes[index])}px`));
    aria();
  }
  function stopDrag() {
    if (!drag) return;
    const previous = drag;
    drag = null;
    doc.body?.classList.remove('is-resizing');
    try { if (previous.handle.hasPointerCapture?.(previous.pointerId)) previous.handle.releasePointerCapture(previous.pointerId); } catch { /* Capture may already have ended. */ }
  }
  function apply(force = false) {
    if (destroyed) return;
    const width = measure();
    const open = openStates();
    const key = `${width}|${open.join(',')}|${media.matches}`;
    if (!force && key === lastKey) return;
    lastKey = key;
    root.dataset.layout = media.matches ? 'stacked' : 'columns';
    if (media.matches) {
      names.forEach(name => root.style.removeProperty(`--${name}-size`));
      sizes = panes.map(pane => Math.max(0, pane.getBoundingClientRect().width));
      aria();
    } else {
      sizes = fitPaneSizes(width, preferred, open);
      paint();
    }
    if (drag) {
      // A details toggle or resize must not release pointer capture. Rebase
      // only when the container/mode really changes, not on our own toggles.
      drag.gesture = createPaneGesture(sizes, open, drag.index);
      drag.startX = drag.lastX;
    }
    keyboard = null;
  }
  function commitGesture(gesture) {
    sizes = gesture.sizes;
    gesture.open.forEach((value, index) => {
      if (panes[index].open !== value) { ownToggles.set(panes[index], value); panes[index].open = value; }
    });
    // Keep closed panes' remembered weight. Rebase only the open panes so that
    // reopening a rail restores its previous share rather than stealing a new one.
    const open = openStates();
    const weight = sum(preferred.filter((_, index) => open[index]));
    const visible = sum(sizes.filter((_, index) => open[index]));
    if (visible > 0) preferred = preferred.map((value, index) => open[index] ? sizes[index] / visible * weight : value);
    lastKey = `${measure()}|${open.join(',')}|${media.matches}`;
    paint();
  }
  function onPointerDown(event) {
    const index = handles.indexOf(event.currentTarget);
    if (event.button !== 0 || event.isPrimary === false || !enabled(index)) return;
    apply();
    event.preventDefault();
    keyboard = null;
    drag = { handle: event.currentTarget, index, pointerId: event.pointerId, startX: event.clientX, lastX: event.clientX,
      gesture: createPaneGesture(sizes, openStates(), index) };
    doc.body?.classList.add('is-resizing');
    try { drag.handle.setPointerCapture(event.pointerId); } catch { /* Document listeners remain a fallback. */ }
  }
  function onPointerMove(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    drag.lastX = event.clientX;
    if (media.matches) return;
    event.preventDefault();
    drag.gesture = updatePaneGesture(drag.gesture, event.clientX - drag.startX);
    commitGesture(drag.gesture);
  }
  function onPointerEnd(event) { if (drag && event.pointerId === drag.pointerId) stopDrag(); }
  function onKeyDown(event) {
    const index = handles.indexOf(event.currentTarget);
    if (!enabled(index)) return;
    const step = event.shiftKey ? 40 : 16;
    const delta = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step
      : event.key === 'Home' ? -Number.MAX_SAFE_INTEGER : event.key === 'End' ? Number.MAX_SAFE_INTEGER : null;
    if (delta === null) return;
    event.preventDefault();
    if (!keyboard || keyboard.index !== index) keyboard = { index, delta: 0, gesture: createPaneGesture(sizes, openStates(), index) };
    keyboard.delta = ['Home', 'End'].includes(event.key) ? delta : keyboard.delta + delta;
    keyboard.gesture = updatePaneGesture(keyboard.gesture, keyboard.delta);
    commitGesture(keyboard.gesture);
    if (['Home', 'End'].includes(event.key)) keyboard = null;
  }
  function onBlur() { keyboard = null; }
  function onToggle(event) {
    if (!panes.includes(event.target)) return;
    if (ownToggles.has(event.target) && ownToggles.get(event.target) === event.target.open) { ownToggles.delete(event.target); return; }
    ownToggles.delete(event.target);
    if (!openStates().some(Boolean)) { ownToggles.set(event.target, true); event.target.open = true; }
    apply();
  }
  function onResize() { apply(); }
  handles.forEach(handle => {
    handle.addEventListener('pointerdown', onPointerDown);
    handle.addEventListener('keydown', onKeyDown);
    handle.addEventListener('blur', onBlur);
    handle.addEventListener('lostpointercapture', onPointerEnd);
  });
  doc.addEventListener('pointermove', onPointerMove, { passive: false });
  doc.addEventListener('pointerup', onPointerEnd);
  doc.addEventListener('pointercancel', onPointerEnd);
  root.addEventListener('toggle', onToggle, true);
  view.addEventListener('resize', onResize);
  media.addEventListener('change', onResize);
  const observer = typeof view.ResizeObserver === 'function' ? new view.ResizeObserver(onResize) : null;
  observer?.observe(root);
  apply(true);

  return {
    reset() {
      stopDrag(); keyboard = null;
      preferred = defaults(measure());
      panes.forEach(pane => { if (!pane.open) { ownToggles.set(pane, true); pane.open = true; } });
      apply(true);
    },
    getSizes() { return [...sizes]; },
    destroy() {
      if (destroyed) return;
      stopDrag();
      destroyed = true;
      observer?.disconnect();
      view.removeEventListener('resize', onResize);
      media.removeEventListener('change', onResize);
      root.removeEventListener('toggle', onToggle, true);
      doc.removeEventListener('pointermove', onPointerMove);
      doc.removeEventListener('pointerup', onPointerEnd);
      doc.removeEventListener('pointercancel', onPointerEnd);
      handles.forEach(handle => {
        handle.removeEventListener('pointerdown', onPointerDown);
        handle.removeEventListener('keydown', onKeyDown);
        handle.removeEventListener('blur', onBlur);
        handle.removeEventListener('lostpointercapture', onPointerEnd);
      });
      names.forEach(name => root.style.removeProperty(`--${name}-size`));
      delete root.dataset.layout;
    },
  };
}
