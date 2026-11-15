// Pure numeric chart model. Raw records stay separate from the display sample.
const finite = value => typeof value === 'number' && Number.isFinite(value);
const clamp01 = value => Math.min(1, Math.max(0, finite(value) ? value : 0));

function extrema(values) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
  return [minimum, maximum];
}

function unitBetween(value, minimum, maximum) {
  if (minimum === maximum) return .5;
  const span = maximum - minimum;
  if (Number.isFinite(span) && span > 0) return clamp01((value - minimum) / span);
  const factor = Math.max(Math.abs(minimum), Math.abs(maximum));
  return factor ? clamp01((value / factor - minimum / factor) / (maximum / factor - minimum / factor)) : .5;
}

function chooseScale(values, requested, loss) {
  if (!values.length) return { scale: 'linear', reason: '暂无有效数值；缺失记录不视为零。' };
  const [minimum, maximum] = extrema(values);
  if (requested === 'log') return minimum > 0
    ? { scale: 'log', reason: '按请求使用正值对数尺度。' }
    : { scale: 'symlog', reason: '包含零或负值，正值 log 不适用，已回退为保留符号的 symlog。' };
  if (requested === 'symlog') return { scale: 'symlog', reason: '按请求使用保留零与正负号的对称对数尺度。' };
  if (requested === 'linear') return { scale: 'linear', reason: '按请求使用线性尺度。' };
  if (requested !== 'auto') return { scale: 'linear', reason: '未识别尺度请求，使用线性尺度。' };
  if (!loss) return { scale: 'linear', reason: '普通指标默认使用线性尺度。' };
  if (minimum > 0 && maximum / minimum >= 100) {
    return { scale: 'log', reason: '所有 Loss 均为正值，数值跨越至少两个数量级。' };
  }
  const nonzero = values.filter(value => value !== 0).map(Math.abs);
  const [smallest, largest] = extrema(nonzero);
  if (minimum <= 0 && ((minimum < 0 && maximum > 0) || (nonzero.length && largest / smallest >= 100))) {
    return { scale: 'symlog', reason: minimum < 0 && maximum > 0
      ? 'Loss 跨越正负与零，使用 symlog 保留符号。'
      : 'Loss 含零或负值且非零绝对值跨越至少两个数量级，使用 symlog。' };
  }
  return { scale: 'linear', reason: '数值跨度较小，使用线性尺度；零值保持为零。' };
}

function sampleSegment(segment, left, innerWidth) {
  const columns = Math.max(1, Math.floor(innerWidth));
  const sampled = [];
  let bucket = -1;
  let first = null, minimum = null, maximum = null, last = null;
  function flush() {
    if (!first) return;
    const choices = [first, minimum, maximum, last].sort((a, b) => a.index - b.index);
    let previous = -1;
    for (const record of choices) if (record.index !== previous) { sampled.push(record); previous = record.index; }
  }
  for (const record of segment) {
    const nextBucket = Math.min(columns - 1, Math.max(0, Math.floor(unitBetween(record.x, left, left + innerWidth) * columns)));
    if (nextBucket !== bucket) {
      flush();
      bucket = nextBucket;
      first = minimum = maximum = last = record;
    } else {
      last = record;
      if (record.value < minimum.value) minimum = record;
      if (record.value > maximum.value) maximum = record;
    }
  }
  flush();
  return sampled;
}

/** Keep source points intact; sample only the line to the available pixel width. */
export function createChartModel(points, key, { width = 280, height = 124, scale = 'auto', loss = false, padding = 8 } = {}) {
  width = finite(width) && width > 0 ? width : 280;
  height = finite(height) && height > 0 ? height : 124;
  padding = Math.min(finite(padding) ? Math.max(0, padding) : 8, width / 2, height / 2);
  const ordered = (Array.isArray(points) ? points : []).map((point, sourceIndex) => ({
    point, sourceIndex, step: finite(point?.steps) ? point.steps : sourceIndex,
    value: finite(point?.[key]) ? point[key] : null,
  })).sort((a, b) => a.step - b.step || a.sourceIndex - b.sourceIndex);
  const values = ordered.filter(record => record.value !== null).map(record => record.value);
  const rawDomain = values.length ? extrema(values) : [null, null];
  const selected = chooseScale(values, scale, loss);
  const maximumAbsolute = values.reduce((value, item) => Math.max(value, Math.abs(item)), 0);
  const constant = maximumAbsolute ? Math.max(Number.MIN_VALUE, maximumAbsolute / 100) : 1;
  const transform = selected.scale === 'log' ? value => Math.log(value)
    : selected.scale === 'symlog' ? value => Math.sign(value) * Math.log1p(Math.abs(value) / constant)
      : value => value;
  const transformedDomain = values.length ? rawDomain.map(transform) : [0, 0];
  const firstStep = ordered[0]?.step ?? 0;
  const lastStep = ordered.at(-1)?.step ?? firstStep;
  const innerWidth = Math.max(0, width - padding * 2);
  const innerHeight = Math.max(0, height - padding * 2);
  const records = ordered.map((record, index) => ({
    point: record.point, index, step: record.step, value: record.value,
    x: padding + unitBetween(record.step, firstStep, lastStep) * innerWidth,
    y: record.value === null ? null
      : padding + (1 - (.06 + .88 * unitBetween(transform(record.value), ...transformedDomain))) * innerHeight,
  }));
  const fullSegments = [];
  let current = [];
  for (const record of records) {
    if (record.value === null) {
      if (current.length) fullSegments.push(current);
      current = [];
    } else current.push(record);
  }
  if (current.length) fullSegments.push(current);
  const segments = fullSegments.map(segment => sampleSegment(segment, padding, innerWidth));

  function nearestAtRatio(ratio) {
    if (!records.length) return null;
    const fraction = clamp01(ratio);
    const target = firstStep * (1 - fraction) + lastStep * fraction;
    let left = 0, right = records.length;
    while (left < right) {
      const middle = (left + right) >>> 1;
      if (records[middle].step < target) left = middle + 1;
      else right = middle;
    }
    if (left === 0) return records[0];
    if (left === records.length) return records.at(-1);
    const before = records[left - 1], after = records[left];
    return target - before.step <= after.step - target ? before : after;
  }
  return {
    width, height, scale: selected.scale,
    scaleLabel: ({ linear: '线性', log: 'Log · 正值', symlog: 'SymLog · 保留符号' })[selected.scale],
    reason: selected.reason, constant, domain: rawDomain,
    records, segments, totalCount: records.length, validCount: values.length,
    sampledCount: segments.reduce((count, segment) => count + segment.length, 0),
    nearestAtRatio,
    recordAtIndex(index) { return Number.isInteger(index) && index >= 0 ? records[index] ?? null : null; },
  };
}

/** Six significant digits; small losses never silently turn into zero. */
export function formatChartValue(value) {
  if (!finite(value)) return '—';
  if (value === 0) return '0';
  return (Math.abs(value) < 1e-3 ? value.toExponential(5) : value.toPrecision(6)).replace('e+', 'e');
}
