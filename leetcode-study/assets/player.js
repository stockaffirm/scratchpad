/* Shared player: question -> code -> flow -> diagram. Flow arrows jump between flow steps; diagram arrows step through every sub-step. */
(function(){
const ORDER = window.ORDER;
const P = window.PROBLEM, $ = s => document.querySelector(s);
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
let vi = 0, idx = 0, timer = null, speed = 1;

function hl(line){
  return esc(line).replace(/(#.*$)|('[^']*'|"[^"]*")|\b(class|def|return|for|in|if|elif|else|while|and|or|not|import|from|reversed|range)\b|\b(\d+)\b/g,
    (m,c,s,k,n)=> c?`<span class="c">${c}</span>`: s?`<span class="s">${s}</span>`: k?`<span class="k">${k}</span>`: `<span class="n">${n}</span>`);
}

const cur = ORDER.findIndex(d=>d.startsWith(P.id+"-")), pv = ORDER[cur-1], nx = ORDER[cur+1];
const pnav = `<div class="pnav bot">${pv?`<a href="../${pv}/">◀ Previous problem</a>`:"<span></span>"}${nx?`<a data-next href="../${nx}/">Next problem ▶</a>`:"<span></span>"}</div>`;
$("#app").innerHTML = `
<div class="pnav"><a href="../">← All problems</a><span class="pos">${cur+1} of ${ORDER.length}</span>${nx?`<a data-next class="go" href="../${nx}/">Next problem ▶</a>`:"<span></span>"}</div>
<h1>${P.id}. ${esc(P.title)}</h1>
<p class="sub"><span class="tag ${P.level}">${P.level}</span><span class="tag">${esc(P.topic)}</span></p>
<section class="sec"><h2><b>1</b>Question</h2>
  <p class="q">${esc(P.question)}</p>
  <div class="io"><div><small>Example input</small>${esc(P.input)}</div><div class="arrow">→</div><div><small>Output</small>${esc(P.output)}</div></div>
  ${P.why?`<p class="why"><b>Why?</b> ${esc(P.why)}</p>`:""}${P.idea?`<p class="idea"><b>💡 Key idea</b> ${esc(P.idea)}</p>`:""}
</section>
<div class="work"><section class="sec codeSec"><h2><b>2</b>Code</h2><div class="tabs" id="tabs"></div><pre class="code" id="code"></pre><div class="meta" id="meta"></div></section>
<div class="right"><section class="sec"><h2><b>3</b>Flow<span class="arrows"><button id="fprev" title="Previous flow step">◀</button><button id="fnext" title="Next flow step">▶</button></span></h2><div class="flow" id="flow"></div></section>
<section class="sec dock"><h2><b>4</b>Diagram<span class="arrows"><button id="dprev" title="Previous detail step (shift+←)">◀</button><span class="dct" id="dct"></span><button id="dnext" title="Next detail step (shift+→)">▶</button></span></h2><div class="stage" id="stage"></div><p class="cap" id="cap"></p></section></div></div>
<section class="sec"><h2>Remember</h2>
  <div class="recap">${esc(P.best)}</div>
  <details><summary>Key notes</summary><ul>${P.notes.map(n=>`<li>${esc(n)}</li>`).join("")}</ul></details></section>
${pnav}
<div class="bar"><div class="in">
  <button id="prev" title="Previous flow step (←)">◀ Flow</button>
  <button id="play" class="play" title="Play / Pause (space)">▶ Play</button>
  <button id="next" title="Next flow step (→)">Flow ▶</button>
  <button id="done" class="dn" title="Mark complete">☐ Done</button>
  <div class="prog"><i id="pg"></i></div><span class="ct" id="ct"></span>
  <select id="spd" title="Speed"><option value="2">0.5×</option><option value="1" selected>1×</option><option value="0.5">2×</option></select>
</div></div>`;

document.querySelectorAll("a[data-next]").forEach(a=>a.addEventListener("click",()=>Progress.set(P.id,true)));
const V = () => P.variants[vi];
function micro(){ const v=V(); if(!v._m){ v._m=[]; v.frames.forEach((f,fi)=>{ if(f.sub&&f.sub.length) f.sub.forEach((m,si)=>v._m.push({fi,si,m,f})); else v._m.push({fi,si:0,m:f,f}); }); } return v._m; }
const M = () => micro()[idx];
const firstOf = fi => micro().findIndex(x=>x.fi===fi);

function renderTabs(){
  $("#tabs").innerHTML = P.variants.length>1 ? P.variants.map((v,i)=>`<button class="${i==vi?"on":""}" data-i="${i}">${esc(v.name)}</button>`).join("") : "";
  $("#tabs").style.display = P.variants.length>1 ? "" : "none";
  $("#tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{stop();vi=+b.dataset.i;idx=0;stage.innerHTML="";rowEls.clear();renderTabs();renderStatic();render();});
}
function renderStatic(){
  $("#code").innerHTML = V().code.split("\n").map((l,i)=>`<span class="ln" data-n="${i+1}"><i>${i+1}</i>${hl(l)||" "}</span>`).join("");
  $("#meta").innerHTML = `<span>Time <b>${esc(V().time)}</b></span><span>Space <b>${esc(V().space)}</b></span><span>${esc(V().explain)}</span>`;
  $("#flow").innerHTML = V().frames.map((f,i)=>`<span class="st" data-i="${i}">${esc(f.title)}</span>`).join('<span class="ar">→</span>');
  $("#flow").querySelectorAll(".st").forEach(s=>s.onclick=()=>{stop();idx=firstOf(+s.dataset.i);render();});
}

/* Diagram renderer: keeps elements between steps so CSS transitions animate (cells recolour, values pop, pointers glide). */
const stage = $("#stage"), rowEls = new Map();
function popEl(el){ el.classList.remove("pop"); void el.offsetWidth; el.classList.add("pop"); }
function setCell(cw, v, hot){
  const el = cw.firstChild, t = String(v), sp = t===" ", nul = t==="·", txt = sp?"␣":t;
  if(el.textContent !== txt){ const had = el.textContent !== ""; el.textContent = txt; if(had) popEl(el); }
  el.classList.toggle("hot", !!hot); el.classList.toggle("sp", sp||nul); el.classList.toggle("nul", nul);
}
function mkCw(){ const cw=document.createElement("span"); cw.className="cw new"; const c=document.createElement("span"); c.className="cell"; cw.appendChild(c); return cw; }
const cwsOf = box => [...box.children].filter(c=>c.classList.contains("cw"));
function syncCells(box, vals, hotFn){
  let cws = cwsOf(box);
  while(cws.length < vals.length){ const c=mkCw(), pt=box.querySelector(".ptr"); box.insertBefore(c, pt); cws.push(c); }
  while(cws.length > vals.length) cws.pop().remove();
  vals.forEach((v,i)=>setCell(cws[i], v, hotFn(i)));
}
function makeRow(r){
  const el=document.createElement("div"); el.className="row rnew";
  const nm=document.createElement("span"); nm.className="nm"; nm.textContent=r.name; el.appendChild(nm);
  const body=document.createElement("div"); body.className = r.kind==="grid" ? "grid2" : "cells"+(r.kind==="list"?" listrow":""); el.appendChild(body);
  return {el, body, kind:r.kind||"", ptrs:new Map()};
}
function placePtrs(st, marks){
  const want = new Map();
  Object.entries(marks||{}).forEach(([i,labels])=>String(labels).split(",").forEach((lb,k)=>want.set(lb,{i:+i,k})));
  st.ptrs.forEach((el,lb)=>{ if(!want.has(lb)){ el.classList.remove("on"); st.ptrs.delete(lb); setTimeout(()=>el.remove(),350); } });
  let maxk=-1;
  want.forEach((w,lb)=>{
    const cw=cwsOf(st.body)[w.i]; if(!cw) return; maxk=Math.max(maxk,w.k);
    let el=st.ptrs.get(lb);
    const left=cw.offsetLeft+cw.offsetWidth/2, top=cw.offsetTop+cw.offsetHeight+3+w.k*15;
    if(!el){ el=document.createElement("span"); el.className="ptr"; el.textContent=lb; el.style.left=left+"px"; el.style.top=top+"px"; st.body.appendChild(el); st.ptrs.set(lb,el); void el.offsetWidth; el.classList.add("on"); }
    else { el.style.left=left+"px"; el.style.top=top+"px"; el.classList.add("on"); }
  });
  st.body.style.paddingBottom = (maxk>=0 ? 18+maxk*15 : 0)+"px";
}
function renderStage(rows, facts){
  const seen=new Set(), order=[];
  rows.forEach(r=>{
    let st=rowEls.get(r.name);
    if(!st || st.kind!==(r.kind||"")){ if(st) st.el.remove(); st=makeRow(r); rowEls.set(r.name,st); }
    seen.add(r.name); order.push(st.el);
    if(r.kind==="grid"){
      while(st.body.children.length < r.cells.length){ const g=document.createElement("div"); g.className="cells"; st.body.appendChild(g); }
      while(st.body.children.length > r.cells.length) st.body.lastChild.remove();
      r.cells.forEach((row,ri)=>syncCells(st.body.children[ri], row, ci=>r.hot.includes(ri+","+ci)));
    } else {
      syncCells(st.body, r.cells, i=>r.hot.includes(i));
      st.body.querySelectorAll(":scope > .cw");
    }
    st.pending = r;
  });
  [...rowEls.keys()].forEach(k=>{ if(!seen.has(k)){ rowEls.get(k).el.remove(); rowEls.delete(k); } });
  // facts row
  let fr = rowEls.get("\u0000facts");
  const entries = facts ? Object.entries(facts) : [];
  if(entries.length){
    if(!fr){ const el=document.createElement("div"); el.className="row rnew"; el.innerHTML='<span class="nm"></span><div class="facts"></div>'; fr={el,body:el.lastChild,kind:"f",ptrs:new Map()}; rowEls.set("\u0000facts",fr); }
    fr.body.innerHTML = entries.map(([k,v])=>`<span class="fc">${esc(k)} = ${esc(v)}</span>`).join(""); order.push(fr.el);
  } else if(fr){ fr.el.remove(); rowEls.delete("\u0000facts"); }
  order.forEach((el,i)=>{ if(stage.children[i]!==el) stage.insertBefore(el, stage.children[i]||null); });
  rows.forEach(r=>{ const st=rowEls.get(r.name); if(st.kind!=="grid") placePtrs(st, r.marks); });
}
function render(){
  const X = M(), f = X.f, m = X.m, fi = X.fi, nf = V().frames.length, mm = micro(), last = mm.length-1;
  const lines = m.codeLines || f.codeLines, rows = m.rows || f.rows, facts = m.facts !== undefined ? m.facts : f.facts;
  document.querySelectorAll(".ln").forEach(el=>el.classList.toggle("on", lines.includes(+el.dataset.n)));
  const first = document.querySelector(".ln.on"); if(first){const box=$("#code"); const t=first.offsetTop-box.clientHeight/3; if(box.scrollHeight>box.clientHeight) box.scrollTo({top:t,behavior:"smooth"});}
  document.querySelectorAll(".st").forEach((s,i)=>{s.classList.toggle("on",i==fi);s.classList.toggle("done",i<fi);});
  renderStage(rows, facts);
  const cap=$("#cap"); if(cap.textContent!==m.description){ cap.textContent = m.description; cap.classList.remove("swap"); void cap.offsetWidth; cap.classList.add("swap"); }
  $("#pg").style.width = ((idx+1)/mm.length*100)+"%"; $("#ct").textContent = `${fi+1} / ${nf}`; $("#dct").textContent = `${idx+1}/${mm.length}`;
  const atStart = idx===0, atEnd = fi===nf-1;
  $("#prev").disabled = $("#fprev").disabled = atStart; $("#next").disabled = $("#fnext").disabled = atEnd;
  $("#dprev").disabled = atStart; $("#dnext").disabled = idx===last;
  if(atEnd && !Progress.has(P.id)) Progress.set(P.id,true);
  updDone();
  $("#play").textContent = timer ? "❚❚ Pause" : (idx===last ? "↺ Replay" : "▶ Play");
}
function updDone(){ const d=Progress.has(P.id), b=$("#done"); b.classList.toggle("on",d); b.textContent=d?"✓ Done":"☐ Done"; }
$("#done").onclick=()=>{Progress.set(P.id,!Progress.has(P.id));updDone();};
function stop(){ if(timer){clearInterval(timer);timer=null;} }
function micro1(d){ const n=micro().length; idx=Math.min(n-1,Math.max(0,idx+d)); render(); }
function flow(d){ const X=M(); if(d>0){ if(X.fi<V().frames.length-1) idx=firstOf(X.fi+1); } else { idx = X.si>0 ? firstOf(X.fi) : firstOf(Math.max(0,X.fi-1)); } render(); }
function play(){
  const n=micro().length;
  if(timer){stop();render();return;}
  if(idx===n-1) idx=0;
  render(); timer=setInterval(()=>{ if(idx>=n-1){stop();render();return;} micro1(1); if(idx>=n-1){stop();render();} }, 1500*speed);
  render();
}
$("#prev").onclick=$("#fprev").onclick=()=>{stop();flow(-1);}; $("#next").onclick=$("#fnext").onclick=()=>{stop();flow(1);};
$("#dprev").onclick=()=>{stop();micro1(-1);}; $("#dnext").onclick=()=>{stop();micro1(1);}; $("#play").onclick=play;
$("#spd").onchange=e=>{speed=+e.target.value; if(timer){stop();play();}};
document.addEventListener("keydown",e=>{ if(e.key=="ArrowRight"){stop();e.shiftKey?micro1(1):flow(1);} else if(e.key=="ArrowLeft"){stop();e.shiftKey?micro1(-1):flow(-1);} else if(e.key==" "){e.preventDefault();play();} });
document.title = `${P.id}. ${P.title}`;
renderTabs(); renderStatic(); render();
})();
