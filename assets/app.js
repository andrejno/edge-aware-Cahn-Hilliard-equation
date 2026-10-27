const D = window.CHPM_DATA;
const $ = id => document.getElementById(id);
const fmt = (x, n = 4) => Number(x).toFixed(n);
const sign = x => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(4)}`;
const sci = x => Number(x).toExponential(2);
const esc = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const familyNames = {connections:'Connections', junctions:'Junctions', multiscale:'Multiscale', noise:'Noisy observations'};
let visibleCases = [];

function options(id, values, selected) {
  $(id).innerHTML = values.map(x => `<option value="${esc(x.value ?? x)}">${esc(x.label ?? x)}</option>`).join('');
  if (selected !== undefined) $(id).value = selected;
}

function table(id, headings, rows, highlight = () => false) {
  $(id).innerHTML = `<table><thead><tr>${headings.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map((row, i) => `<tr${highlight(row, i) ? ' class="highlight"' : ''}>${row.map((v, j) => `<${j === 0 ? 'th scope="row"' : 'td'}>${v}</${j === 0 ? 'th' : 'td'}>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function changeDataset() {
  const dataset = $('case-dataset').value;
  visibleCases = D.cases.filter(c => c.dataset === dataset);
  options('case-select', visibleCases.map(c => ({value:c.id,label:c.id.replace('test_retina_', '').replace('test_', '')})), dataset === 'synthetic' ? 'test_junctions_02' : 'test_retina_im0163_c0_w4');
  renderCase();
}

function renderCase() {
  const c = visibleCases.find(x => x.id === $('case-select').value);
  const left = $('method-left').value, right = $('method-right').value;
  const path = `assets/images/${c.id}`;
  $('observed-image').src = `${path}_observed.png`;
  $('reference-image').src = `${path}_reference.png`;
  $('reference-caption').textContent = c.dataset === 'retina' ? 'AH reference' : 'Reference';
  for (const [side, method] of [['left', left], ['right', right]]) {
    $(`image-${side}`).src = `${path}_${method}.png`;
    $(`image-${side}`).alt = `${method} reconstruction of ${c.id.replace('test_', '')}`;
    $(`error-${side}`).src = `${path}_${method}_error.png`;
    $(`${side}-tag`).textContent = method;
  }
  $('mask-image').src = `${path}_mask.svg`;
  $('case-title').textContent = c.id.replace('test_retina_', '').replace('test_', '');
  const leftScore = c.scores[left], rightScore = c.scores[right];
  $('case-scores').innerHTML = [[left,leftScore],[right,rightScore]].map(([m,s]) => `<div class="case-score"><div><span>${esc(m)}</span><strong${m === 'CHPM' ? ' class="teal"' : ''}>${fmt(s.masked_dice)}</strong></div><p>clDice ${fmt(s.masked_cldice)} · RMSE ${fmt(s.masked_rmse)}</p></div>`).join('') + `<div class="score-difference">${esc(right)} − ${esc(left)} <strong>${sign(rightScore.masked_dice - leftScore.masked_dice)}</strong></div>`;
  $('case-detail').textContent = c.dataset === 'retina' ? `192 × 192 pixels · ${c.metadata.width}-pixel scratches` : `96 × 96 pixels · ${familyNames[c.family]}`;
  $('download-field').href = `data/fields/${c.dataset}/${right}/${c.id}.npz`;
  $('compare-range').value = 50;
  setDivider();
}

function setDivider() {
  const value = $('compare-range').value;
  $('left-layer').style.clipPath = `inset(0 ${100 - value}% 0 0)`;
  $('divider').style.left = `${value}%`;
  $('compare-range').setAttribute('aria-valuetext', `${value}% left method`);
}

function moveCase(direction) {
  const index = visibleCases.findIndex(c => c.id === $('case-select').value);
  $('case-select').value = visibleCases[(index + direction + visibleCases.length) % visibleCases.length].id;
  renderCase();
}

function renderFamilies() {
  const comparator = $('family-comparator').value;
  $('family-chart').innerHTML = Object.keys(familyNames).map(family => {
    const a = D.families.find(r => r.family === family && r.method === 'CHPM').masked_dice;
    const b = D.families.find(r => r.family === family && r.method === comparator).masked_dice;
    return `<div class="family-row"><div class="family-row-head"><span>${familyNames[family]}</span><strong>${sign(a-b)}</strong></div><div class="bar-track"><div class="bar chpm" style="width:${100*a}%"><span class="bar-value">${fmt(a)}</span></div></div><div class="bar-track"><div class="bar" style="width:${100*b}%;background:${D.colors[comparator]}"><span class="bar-value${b < .13 ? ' outside' : ''}">${fmt(b)}</span></div></div></div>`;
  }).join('');
  $('comparator-label').textContent = comparator;
  $('comparator-key').style.background = D.colors[comparator];
}

function renderSummary() {
  const dataset = $('summary-dataset').value, rows = D.summary[dataset];
  const metrics = ['masked_dice', 'masked_cldice', 'masked_rmse'];
  const best = metrics.map(m => m === 'masked_rmse' ? Math.min(...rows.map(r => r[m])) : Math.max(...rows.map(r => r[m])));
  table('summary-table', ['Method','Dice','clDice','RMSE'], rows.map(r => [r.method, ...metrics.map((m,i) => Math.abs(r[m]-best[i]) < 1e-10 ? `<span class="best">${fmt(r[m])}</span>` : fmt(r[m]))]), row => row[0] === 'CHPM');
  $('summary-note').textContent = dataset === 'synthetic' ? '24 synthetic test cases · outcomes inside the missing region' : '14 source-image means · 4 crop–mask cases per image';
}

function svg(id, height, content, width) {
  const w = width || Math.max(320, $(id).clientWidth);
  $(id).innerHTML = `<svg viewBox="0 0 ${w} ${height}" role="img" aria-label="${esc(id.replaceAll('-', ' '))}">${content}</svg>`;
}

function text(x,y,value,anchor='start',cls='') {
  return `<text x="${x}" y="${y}" text-anchor="${anchor}"${cls ? ` class="${cls}"` : ''}>${esc(value)}</text>`;
}

function line(x1,y1,x2,y2,color='#c7d1cc',extra='') {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
}

function renderContrasts() {
  const rows = D.contrasts[$('contrast-dataset').value], w = Math.max(320, $('contrast-chart').clientWidth), h = 220;
  const left = w < 500 ? 85 : 115, right = w < 500 ? 65 : 100;
  const lo = -.2, hi = .75, x = v => left + (v-lo)/(hi-lo)*(w-left-right);
  let s = '';
  [-.2,0,.2,.4,.6].forEach(t => {s += line(x(t),15,x(t),182,t === 0 ? '#94a8a1' : '#dce3dc',t === 0 ? 'stroke-dasharray="4 4"' : '');s += text(x(t),205,t === 0 ? '0' : t.toFixed(1),'middle','numeric');});
  rows.forEach((r,i) => {
    const y = 28 + i*33;
    s += text(0,y+4,r.method_b);
    s += line(x(r.ci_low),y,x(r.ci_high),y,'#72978c','stroke-width="2"');
    s += line(x(r.ci_low),y-4,x(r.ci_low),y+4,'#72978c') + line(x(r.ci_high),y-4,x(r.ci_high),y+4,'#72978c');
    s += `<circle cx="${x(r.mean_difference)}" cy="${y}" r="4" fill="#167367"/>`;
    s += text(w,y+4,sign(r.mean_difference),'end','numeric');
  });
  svg('contrast-chart',h,s,w);
}

function renderComplete() {
  const dataset = $('complete-dataset').value, metric = $('complete-metric').value;
  const cases = D.cases.filter(c => c.dataset === dataset);
  table('complete-table',['Case',...D.methods],cases.map(c => [c.id.replace('test_retina_', '').replace('test_', ''),...D.methods.map(m => fmt(c.scores[m][metric]))]));
}

function renderCoefficient() {
  const gamma = Number($('gamma').value), k = Number($('edge-k').value);
  $('gamma-value').textContent = fmt(gamma,2);$('k-value').textContent = fmt(k,2);
  const w = Math.max(320,$('coefficient-chart').clientWidth), h=208, left=38, right=13, top=20, bottom=40;
  const x=v=>left+v/1.2*(w-left-right),y=v=>h-bottom-v*(h-top-bottom);
  let s='';
  [0,.5,1].forEach(t=>{s+=line(left,y(t),w-right,y(t),'#d3ddd5');s+=text(left-8,y(t)+4,String(t),'end','numeric');});
  [0,.3,.6,.9,1.2].forEach(t=>s+=text(x(t),h-bottom+20,t.toFixed(1),'middle','numeric'));
  const points=Array.from({length:121},(_,i)=>{const g=i/100;return `${x(g)},${y(gamma+(1-gamma)/(1+(g/k)**2))}`;}).join(' ');
  s+=line(x(0),y(1),x(1.2),y(1),'#7c8990','stroke-dasharray="5 4"');
  s+=`<polyline points="${points}" fill="none" stroke="#167367" stroke-width="2.4"/>`;
  s+=text(left,12,'a[u]')+text(w/2,h-1,'Smoothed gradient magnitude','middle')+text(w-right,y(1)-7,'CH','end','plot-legend');
  svg('coefficient-chart',h,s,w);
}

function renderProfile() {
  const dataset=$('profile-dataset').value, c=D.calibration[dataset], w=Math.max(320,$('profile-chart').clientWidth),h=196;
  const x=v=>38+(v+7)/14*(w-52),y=v=>145-(v+1)/2*120;
  let s='';
  [-1,0,1].forEach(v=>{s+=line(38,y(v),w-14,y(v),'#dce3dc');s+=text(28,y(v)+4,String(v),'end','numeric');});
  [-6,-3,0,3,6].forEach(v=>s+=text(x(v),166,String(v),'middle','numeric'));
  for(const [key,color,dash] of [['target','#167367',''],['matched','#355b91','stroke-dasharray="4 3"']]){
    const points=c.profiles.x.map((v,i)=>Math.abs(v)<=7?`${x(v)},${y(c.profiles[key][i])}`:null).filter(Boolean).join(' ');
    s+=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" ${dash}/>`;
  }
  s+=text(w/2,190,'Distance from interface (pixels)','middle')+text(48,17,'CHPM', 'start','plot-legend')+text(110,17,'CH-width','start','plot-legend');
  svg('profile-chart',h,s,w);
  $('profile-note').textContent=`a₀ = ${fmt(c.constant_a,7)} · width ${fmt(c.target_width,4)} pixels · mismatch ${sci(c.absolute_width_error)} pixels`;
}

function renderControls() {
  table('coefficient-table',['Method','Synthetic Dice','Retinal Dice'],['CHPM','CH-width','Frozen'].map(m=>[m,...['synthetic','retina'].map(d=>fmt(D.summary[d].find(r=>r.method===m).masked_dice))]),r=>r[0]==='CHPM');
  const noise=D.cases.filter(c=>c.dataset==='synthetic'&&c.family==='noise');
  table('noise-table',['Method','Reference RMSE'],D.methods.map(m=>[m,fmt(noise.reduce((s,c)=>s+c.scores[m].outside_rmse,0)/noise.length)]),r=>r[0]==='CHPM');
}

function renderRetina() {
  const reference=$('retina-reference').value,key=reference==='AH'?'scores':'vk_scores';
  const cases=D.cases.filter(c=>c.dataset==='retina');
  const ids=[...new Set(cases.map(c=>c.metadata.image_id))].sort();
  const w=Math.max(320,$('retina-chart').clientWidth),h=370,x=v=>65+v*(w-83);
  let s='';
  [0,.25,.5,.75,1].forEach(t=>{s+=line(x(t),10,x(t),330,'#dce3dc');s+=text(x(t),349,String(t),'middle','numeric');});
  ids.forEach((id,i)=>{
    const group=cases.filter(c=>c.metadata.image_id===id),y=20+i*23;
    const a=group.reduce((n,c)=>n+c[key].CH.masked_dice,0)/4,b=group.reduce((n,c)=>n+c[key].CHPM.masked_dice,0)/4;
    s+=text(0,y+4,id)+line(x(a),y,x(b),y,'#adbbb4','stroke-width="1.5"');
    s+=`<circle cx="${x(a)}" cy="${y}" r="3.5" fill="#7c8990"/><circle cx="${x(b)}" cy="${y}" r="4" fill="#167367"/>`;
  });
  s+=text(w/2,369,'Mean masked Dice · CH (gray), CHPM (green)','middle');
  svg('retina-chart',h,s,w);
  const ww=Math.max(320,$('width-chart').clientWidth),hh=241,xx=v=>v===4?ww*.32:ww*.76,yy=v=>175-v*150;
  s='';
  [0,.25,.5,.75,1].forEach(v=>{s+=line(34,yy(v),ww-12,yy(v),'#dce3dc');s+=text(27,yy(v)+4,String(v),'end','numeric');});
  [4,8].forEach(width=>{s+=text(xx(width),195,`${width} pixels`,'middle');D.methods.forEach((m,i)=>{const v=D.widths[reference][String(width)].find(r=>r.method===m).masked_dice;s+=`<circle cx="${xx(width)+(i-2.5)*10}" cy="${yy(v)}" r="4" fill="${D.colors[m]}"/>`;});});
  D.methods.forEach((m,i)=>{const lx=15+(i%3)*(ww/3),ly=216+Math.floor(i/3)*20;s+=`<rect x="${lx}" y="${ly-6}" width="7" height="7" fill="${D.colors[m]}"/>`+text(lx+12,ly+1,m,'start','plot-legend');});
  svg('width-chart',hh,s,ww);
}

function renderRefinement() {
  const [dataset,kind]=$('refinement-select').value.split('-');
  const rows=D.refinement.filter(r=>r.dataset===dataset&&r.kind===kind);
  table('refinement-table',['Case','Method',kind==='time'?'Dice · Δt 0.1':'Dice · h 1',kind==='time'?'Dice · Δt 0.05':'Dice · h 0.5','Changed labels (%)','RMS'],rows.map(r=>[r.case_id.replace('test_retina_', '').replace('test_', ''),r.method,fmt(r.original_dice),fmt(r.refined_dice),fmt(r.masked_threshold_disagreement*100,3),sci(r.field_rms)]));
  $('refinement-note').textContent=kind==='grid'?'Δt = 0.05 · Fine fields averaged over 2 × 2 blocks onto the original grid':'h = 1 · Selected parameters and final times';
}

function renderNumericalTables() {
  table('parameters-table',['Dataset','Method','τ','γ','K','T'],D.parameters.map(r=>[r.Dataset==='synthetic'?'Synthetic':'Retinal',r.Method,fmt(r.tau,2),r.Method==='CH'?'1':fmt(r.gamma,1),r.Method==='CH'?'—':fmt(r.K,2),r.T]));
  table('tuning-table',['Dataset','Method','Rank','τ','γ','K','T','Dice','RMSE'],D.tuning.map(r=>[r.Dataset,r.Method,r.Rank,fmt(r.tau,2),r.Method==='CH'?'—':fmt(r.gamma,1),r.Method==='CH'?'—':fmt(r.K,2),r.T,fmt(r.Dice),fmt(r.RMSE)]));
  table('diagnostics-table',['Dataset','Method','Reference RMSE','Observation RMSE','Maximum overshoot'],D.diagnostics.map(r=>[r.Dataset,r.Method,fmt(r['Reference RMSE']),fmt(r['Observation RMSE']),r['Maximum overshoot']<.0001?sci(r['Maximum overshoot']):fmt(r['Maximum overshoot'])]));
  table('tv-table',['Dataset','Runs','At tolerance','At 10,000','Maximum relative gap','Divergence residual'],D.tv.map(r=>[r.Dataset,r.Reconstructions,r['At tolerance'],r['At iteration limit'],sci(r['Maximum relative gap']),sci(r['Maximum divergence residual'])]));
  const extended=[];
  for(const key of ['sensitive_time_summary','retina_sensitive_time_summary']){
    const v=D.extended[key];v.comparisons.forEach(r=>extended.push([v.case_id.replace('test_retina_','').replace('test_',''),'CHPM',`${r.original_dt} → ${r.refined_dt}`,fmt(r.original_dice),fmt(r.refined_dice),sci(r.field_rms)]));
  }
  const f=D.extended.fine_grid_time_summary;
  extended.push(['multiscale_04 · h 0.5','CHPM',`${f.original_dt} → ${f.refined_dt}`,fmt(f.original_dice),fmt(f.refined_dice),sci(f.field_rms)]);
  table('extended-table',['Case','Method','Δt','Original Dice','Refined Dice','RMS'],extended);
  table('spatial-table',['Cells N','L² error','Order'],D.spatial.map(r=>[r.N,sci(r.L2_error),r.observed_order===null?'—':fmt(r.observed_order,2)]));
  table('temporal-table',['Δt','RMS error','Order'],D.temporal.map(r=>[r.dt,sci(r.rms_error),r.order===null?'—':fmt(r.order,2)]));
}

function logPlot(id,rows,xkey,ykey,xlabel) {
  const w=Math.max(320,$(id).clientWidth),h=220,left=60,right=15,top=20,bottom=45;
  const xs=rows.map(r=>Math.log10(r[xkey])),ys=rows.map(r=>Math.log10(r[ykey]));
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.floor(Math.min(...ys)),maxY=Math.ceil(Math.max(...ys));
  const x=v=>left+(Math.log10(v)-minX)/(maxX-minX)*(w-left-right),y=v=>h-bottom-(Math.log10(v)-minY)/(maxY-minY)*(h-top-bottom);
  let s='';
  for(let i=minY;i<=maxY;i++){s+=line(left,y(10**i),w-right,y(10**i),'#dce3dc');s+=text(left-8,y(10**i)+4,`10${['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹'][Math.abs(i)] ? (i<0?'⁻':'')+String(Math.abs(i)).split('').map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]).join('') : '^'+i}`,'end','numeric');}
  rows.filter((_,i)=>rows.length<=4||i%2===1).forEach(r=>s+=text(x(r[xkey]),h-bottom+19,String(r[xkey]),'middle','numeric'));
  const pts=rows.map(r=>`${x(r[xkey])},${y(r[ykey])}`).join(' ');
  s+=`<polyline points="${pts}" fill="none" stroke="#167367" stroke-width="2"/>`;
  rows.forEach(r=>s+=`<circle cx="${x(r[xkey])}" cy="${y(r[ykey])}" r="3.5" fill="#167367"/>`);
  s+=text(w/2,h-4,xlabel,'middle');
  svg(id,h,s,w);
}

function renderPlots() {
  renderContrasts();renderCoefficient();renderProfile();renderRetina();
  logPlot('spatial-chart',D.spatial,'N','L2_error','Cells N · T = 0.05');
  logPlot('temporal-chart',D.temporal,'dt','rms_error','Time step · 12 × 12 grid · T = 2');
}

options('method-left',D.methods,'CH');options('method-right',D.methods,'CHPM');options('family-comparator',D.methods.filter(m=>m!=='CHPM'),'CH');
$('case-dataset').addEventListener('change',changeDataset);
['case-select','method-left','method-right'].forEach(id=>$(id).addEventListener('change',renderCase));
$('compare-range').addEventListener('input',setDivider);
$('previous-case').addEventListener('click',()=>moveCase(-1));$('next-case').addEventListener('click',()=>moveCase(1));
$('show-mask').addEventListener('change',()=>{$('mask-image').style.display=$('show-mask').checked?'':'none';});
$('show-errors').addEventListener('change',()=>{$('comparison').classList.toggle('show-errors',$('show-errors').checked);$('error-legend').hidden=!$('show-errors').checked;});
$('family-comparator').addEventListener('change',renderFamilies);$('summary-dataset').addEventListener('change',renderSummary);$('contrast-dataset').addEventListener('change',renderContrasts);
['complete-dataset','complete-metric'].forEach(id=>$(id).addEventListener('change',renderComplete));
['gamma','edge-k'].forEach(id=>$(id).addEventListener('input',renderCoefficient));$('profile-dataset').addEventListener('change',renderProfile);$('retina-reference').addEventListener('change',renderRetina);$('refinement-select').addEventListener('change',renderRefinement);
document.querySelectorAll('details').forEach(el=>el.addEventListener('toggle',()=>{if(el.open)renderPlots();}));
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderPlots,120);});
changeDataset();renderFamilies();renderSummary();renderComplete();renderControls();renderNumericalTables();renderRefinement();renderPlots();
