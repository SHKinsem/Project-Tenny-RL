import { createChartModel, formatChartValue } from './chart-model.mjs';

const NS = 'http://www.w3.org/2000/svg';
let chartSequence = 0;
const h = (tag, className, text) => { const node=document.createElement(tag); if(className)node.className=className; if(text!==undefined)node.textContent=text; return node; };
const svgNode = (tag, attrs={}) => { const node=document.createElementNS(NS,tag); for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value)); return node; };
const stepText = value => Number.isFinite(value) ? value.toLocaleString('zh-CN',{maximumFractionDigits:0}) : '—';
const timeText = value => Number.isFinite(value) ? new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Hong_Kong',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(value) : '时间未记录';

/** Axis-free plot. Rendered vertices may be reduced; inspection always uses raw records. */
export function createInteractiveChart(container, {key,label,loss=false,series='duration'}) {
  const id=`chart-inspector-${++chartSequence}`;
  const reduce=matchMedia('(prefers-reduced-motion: reduce)');
  let points=[], model=null, requestedScale='auto', activeIndex=null, pointerInside=false, frame=0, disposed=false;
  let tooltipMotion=null, tooltipClosing=false;
  const oldTabIndex=container.getAttribute('tabindex');
  container.replaceChildren(); container.classList.add('chart-interactive'); container.tabIndex=0;
  container.setAttribute('role','group');
  container.setAttribute('aria-label',`${label}。悬停或触摸查看原值；方向键逐点，Home和End到首尾。`);
  const controls=h('div','chart-controls');
  const select=h('select','chart-scale'); select.setAttribute('aria-label',`${label}尺度`);
  for(const [value,text] of [['auto','自动尺度'],['linear','线性'],['log','对数'],['symlog','对称对数']]) { const option=h('option','',text); option.value=value; select.append(option); }
  const scaleInfo=h('span','chart-scale-info'); controls.append(scaleInfo,select);
  const stage=h('div','chart-stage');
  const svg=svgNode('svg',{role:'img','aria-label':`${label}趋势`});
  const path=svgNode('path',{class:`trend-line ${series==='duration'?'':`${series}-series`}`,'vector-effect':'non-scaling-stroke'});
  const empty=svgNode('text',{class:'empty-chart-text','text-anchor':'middle'}); empty.textContent='暂无有效记录';
  const cursor=svgNode('line',{class:'chart-cursor',hidden:''});
  const dot=svgNode('circle',{class:'chart-active-point',r:2.5,hidden:''});
  svg.append(path,empty,cursor,dot); stage.append(svg);
  const tooltip=h('div','chart-tooltip'); tooltip.id=id; tooltip.setAttribute('role','tooltip'); tooltip.hidden=true;
  const valueRow=h('strong','chart-tooltip-value'), stepRow=h('span','chart-tooltip-step'), stampRow=h('span','chart-tooltip-time'), contextRow=h('span','chart-tooltip-context');
  const metadata=h('div','chart-tooltip-meta'); metadata.append(stepRow,stampRow);
  tooltip.append(valueRow,metadata,contextRow);
  const help=h('span','visually-hidden','方向键逐点，PageUp和PageDown跳段，Home和End到首尾，Esc收起。'); help.id=`${id}-help`;
  const live=h('span','visually-hidden'); live.setAttribute('role','status'); live.setAttribute('aria-live','polite');
  container.setAttribute('aria-describedby',help.id); container.append(stage,controls,tooltip,help,live);

  function hideInspector(immediate=false) {
    cursor.setAttribute('hidden',''); dot.setAttribute('hidden','');
    container.classList.remove('is-inspecting'); container.removeAttribute('data-active-index');
    container.setAttribute('aria-describedby',help.id);
    live.textContent='';
    if(immediate || reduce.matches || !tooltip.animate) { tooltipMotion?.cancel(); tooltipMotion=null; tooltipClosing=false; tooltip.hidden=true; return; }
    if(tooltip.hidden || tooltipClosing)return;
    tooltipMotion?.cancel(); tooltipClosing=true;
    const motion=tooltip.animate([{opacity:1},{opacity:0}],{duration:90,easing:'ease-out',fill:'forwards'});
    tooltipMotion=motion;
    motion.finished.then(()=>{if(tooltipMotion===motion){tooltip.hidden=true;tooltipClosing=false;tooltipMotion=null;motion.cancel();}}).catch(()=>{});
  }
  function inspect(record, announce=false) {
    if(!record || !model)return;
    const entering=tooltip.hidden || tooltipClosing;
    if(entering) { tooltipMotion?.cancel(); tooltipMotion=null; tooltipClosing=false; }
    activeIndex=record.index;
    container.dataset.activeIndex=String(record.index);
    container.classList.add('is-inspecting');
    cursor.removeAttribute('hidden'); cursor.setAttribute('x1',record.x); cursor.setAttribute('x2',record.x); cursor.setAttribute('y1',6); cursor.setAttribute('y2',model.height-6);
    if(Number.isFinite(record.y)) { dot.removeAttribute('hidden'); dot.setAttribute('cx',record.x); dot.setAttribute('cy',record.y); } else dot.setAttribute('hidden','');
    const raw=Number.isFinite(record.value)?formatChartValue(record.value):'缺失';
    valueRow.textContent=`${raw}${key==='episodeSeconds'&&Number.isFinite(record.value)?' 秒':''}`;
    stepRow.textContent=`${stepText(record.point.steps)} 步`;
    stampRow.textContent=timeText(record.point.at);
    contextRow.textContent=model.scaleLabel;
    contextRow.hidden=model.scale==='linear';
    tooltip.hidden=false;
    const tipWidth=tooltip.offsetWidth || 180;
    const left=Math.max(4,Math.min(model.width-tipWidth-4,record.x+12));
    const top=stage.offsetTop+Math.max(4,Math.min(model.height-tooltip.offsetHeight-4,(Number.isFinite(record.y)?record.y:30)-tooltip.offsetHeight-10));
    tooltip.style.left=`${left}px`; tooltip.style.top=`${top}px`;
    if(entering && !reduce.matches && tooltip.animate) tooltipMotion=tooltip.animate([{opacity:0,transform:'translateY(3px) scale(.98)'},{opacity:1,transform:'translateY(0) scale(1)'}],{duration:140,easing:'cubic-bezier(.22,1,.36,1)'});
    container.setAttribute('aria-describedby',`${help.id} ${id}`);
    if(announce)live.textContent=`${label} ${valueRow.textContent}，本轮 ${stepText(record.point.steps)} 环境步，${model.scaleLabel}`;
  }
  function draw(animate=false) {
    if(disposed)return;
    const width=Math.max(120,container.clientWidth || 264), height=124;
    model=createChartModel(points,key,{width,height,scale:requestedScale,loss,padding:8});
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    let d='';
    for(const segment of model.segments) for(const [index,record] of segment.entries()) d+=`${index?'L':'M'}${record.x.toFixed(2)},${record.y.toFixed(2)} `;
    path.setAttribute('d',d);
    empty.setAttribute('x',width/2); empty.setAttribute('y',height/2); empty.toggleAttribute('hidden',model.validCount>0);
    scaleInfo.textContent=model.scaleLabel;
    const range=model.validCount?`${formatChartValue(model.domain[0])} ～ ${formatChartValue(model.domain[1])}`:'无有效值';
    const scaleDescription=`${model.scaleLabel} · 范围 ${range}${model.reason?` · ${model.reason}`:''}`;
    scaleInfo.title=scaleDescription; select.title=scaleDescription; select.setAttribute('aria-description',scaleDescription);
    const logOption=select.querySelector('[value="log"]');
    const nonPositive=model.records.some(record=>Number.isFinite(record.value)&&record.value<=0);
    logOption.disabled=nonPositive; logOption.textContent=nonPositive?'对数·需正值':'对数';
    if(requestedScale==='log'&&nonPositive) { requestedScale='symlog'; select.value='symlog'; }
    container.dataset.scale=model.scale; container.dataset.pointCount=String(model.totalCount); container.dataset.drawnCount=String(model.sampledCount);
    if(activeIndex!==null&&!tooltip.hidden)inspect(model.recordAtIndex(activeIndex)); else hideInspector();
    if(animate&&!reduce.matches)path.animate([{opacity:.3},{opacity:1}],{duration:180,easing:'ease-out'});
  }
  function resize() { if(!frame)frame=requestAnimationFrame(()=>{frame=0;draw(false);}); }
  function fromPointer(event,announce=false) {
    if(!model||event.target.closest('select'))return;
    const box=stage.getBoundingClientRect();
    const ratio=Math.max(0,Math.min(1,(event.clientX-box.left-8)/Math.max(1,box.width-16)));
    inspect(model.nearestAtRatio(ratio),announce);
  }
  function onMove(event) { pointerInside=true; fromPointer(event); }
  function onLeave() { pointerInside=false; if(!container.contains(document.activeElement))hideInspector(); }
  function onDown(event) { if(event.target.closest('select'))return; container.focus({preventScroll:true}); fromPointer(event,true); }
  function onFocus(event) { if(event.target===container&&model?.records.length)inspect(model.recordAtIndex(activeIndex??model.records.length-1),true); }
  function onFocusOut(event) { if(!container.contains(event.relatedTarget)&&!pointerInside)hideInspector(); }
  function onKey(event) {
    if(event.target===select||!model?.records.length)return;
    if(event.key==='Escape') { hideInspector(); return; }
    const current=activeIndex??model.records.length-1, jump=Math.max(1,Math.round(model.records.length/10));
    const next=event.key==='ArrowLeft'?current-1:event.key==='ArrowRight'?current+1:event.key==='PageUp'?current-jump:event.key==='PageDown'?current+jump:event.key==='Home'?0:event.key==='End'?model.records.length-1:null;
    if(next===null)return; event.preventDefault(); inspect(model.recordAtIndex(Math.max(0,Math.min(model.records.length-1,next))),true);
  }
  function onScaleChange() { requestedScale=select.value; draw(true); if(model?.records.length)inspect(model.recordAtIndex(activeIndex??model.records.length-1),true); }
  stage.addEventListener('pointermove',onMove); container.addEventListener('pointerleave',onLeave); stage.addEventListener('pointerdown',onDown);
  container.addEventListener('focusin',onFocus); container.addEventListener('focusout',onFocusOut); container.addEventListener('keydown',onKey); select.addEventListener('change',onScaleChange);
  const observer=new ResizeObserver(resize); observer.observe(container);
  return {
    setData(next) { points=next; activeIndex=null; hideInspector(true); draw(true); },
    getModel:()=>model,
    destroy() { disposed=true; tooltipMotion?.cancel(); observer.disconnect(); if(frame)cancelAnimationFrame(frame); stage.removeEventListener('pointermove',onMove); container.removeEventListener('pointerleave',onLeave); stage.removeEventListener('pointerdown',onDown); container.removeEventListener('focusin',onFocus); container.removeEventListener('focusout',onFocusOut); container.removeEventListener('keydown',onKey); select.removeEventListener('change',onScaleChange); if(oldTabIndex===null)container.removeAttribute('tabindex');else container.setAttribute('tabindex',oldTabIndex); container.replaceChildren(); },
  };
}
