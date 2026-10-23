/* Shared player: renders window.PROBLEM (question -> code -> flow -> diagram) with Prev / Play / Next. */
(function(){
const P = window.PROBLEM, $ = s => document.querySelector(s);
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
let vi = 0, fi = 0, timer = null, speed = 1;

function hl(line){
  return esc(line).replace(/(#.*$)|('[^']*'|"[^"]*")|\b(class|def|return|for|in|if|elif|else|while|and|or|not|import|from|reversed|range)\b|\b(\d+)\b/g,
    (m,c,s,k,n)=> c?`<span class="c">${c}</span>`: s?`<span class="s">${s}</span>`: k?`<span class="k">${k}</span>`: `<span class="n">${n}</span>`);
}

$("#app").innerHTML = `
<a href="../">← All problems</a>
<h1>${P.id}. ${esc(P.title)}</h1>
<p class="sub"><span class="tag ${P.level}">${P.level}</span><span class="tag">${esc(P.topic)}</span></p>
<section class="sec"><h2><b>1</b>Question</h2>
  <p class="q">${esc(P.question)}</p>
  <div class="io"><div><small>Input</small>${esc(P.input)}</div><div class="arrow">→</div><div><small>Output</small>${esc(P.output)}</div></div>
</section>
<div class="work"><section class="sec codeSec"><h2><b>2</b>Code</h2><div class="tabs" id="tabs"></div><pre class="code" id="code"></pre><div class="meta" id="meta"></div></section>
<div class="right"><section class="sec"><h2><b>3</b>Flow</h2><div class="flow" id="flow"></div></section>
<section class="sec dock"><h2><b>4</b>Diagram</h2><div class="stage" id="stage"></div><p class="cap" id="cap"></p></section></div></div>
<section class="sec"><h2>Remember</h2>
  <div class="recap">${esc(P.best)}</div>
  <details><summary>Key notes</summary><ul>${P.notes.map(n=>`<li>${esc(n)}</li>`).join("")}</ul></details></section>
<div class="bar"><div class="in">
  <button id="prev" title="Previous (←)">◀ Prev</button>
  <button id="play" class="play" title="Play / Pause (space)">▶ Play</button>
  <button id="next" title="Next (→)">Next ▶</button>
  <div class="prog"><i id="pg"></i></div><span class="ct" id="ct"></span>
  <select id="spd" title="Speed"><option value="2">0.5×</option><option value="1" selected>1×</option><option value="0.5">2×</option></select>
</div></div>`;

const V = () => P.variants[vi], F = () => V().frames[fi];

function renderTabs(){
  $("#tabs").innerHTML = P.variants.length>1 ? P.variants.map((v,i)=>`<button class="${i==vi?"on":""}" data-i="${i}">${esc(v.name)}</button>`).join("") : "";
  $("#tabs").style.display = P.variants.length>1 ? "" : "none";
  $("#tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{stop();vi=+b.dataset.i;fi=0;renderTabs();renderStatic();render();});
}
function renderStatic(){
  $("#code").innerHTML = V().code.split("\n").map((l,i)=>`<span class="ln" data-n="${i+1}"><i>${i+1}</i>${hl(l)||" "}</span>`).join("");
  $("#meta").innerHTML = `<span>Time <b>${esc(V().time)}</b></span><span>Space <b>${esc(V().space)}</b></span><span>${esc(V().explain)}</span>`;
  $("#flow").innerHTML = V().frames.map((f,i)=>`<span class="st" data-i="${i}">${esc(f.title)}</span>`).join('<span class="ar">→</span>');
  $("#flow").querySelectorAll(".st").forEach(s=>s.onclick=()=>{stop();fi=+s.dataset.i;render();});
}
let prevSig = {};
function render(){
  const f = F(), n = V().frames.length;
  document.querySelectorAll(".ln").forEach(el=>el.classList.toggle("on", f.codeLines.includes(+el.dataset.n)));
  const first = document.querySelector(".ln.on"); if(first){const box=$("#code"); const t=first.offsetTop-box.clientHeight/3; if(box.scrollHeight>box.clientHeight) box.scrollTo({top:t,behavior:"smooth"});}
  document.querySelectorAll(".st").forEach((s,i)=>{s.classList.toggle("on",i==fi);s.classList.toggle("done",i<fi);});
  const sig = {};
  $("#stage").innerHTML = f.rows.map(r=>{
    const old = prevSig[r.name];
    return `<div class="row"><span class="nm">${esc(r.name)}</span><div class="cells">${r.cells.map((c,i)=>{
      const t = String(c), sp = t===" ";
      return `<span class="cell${r.hot.includes(i)?" hot":""}${sp?" sp":""}${old&&old[i]!==t?" pop":""}">${sp?"␣":esc(t)}</span>`;}).join("")}</div></div>`;
  }).join("") + (f.facts&&Object.keys(f.facts).length?`<div class="row"><span class="nm"></span><div class="facts">${Object.entries(f.facts).map(([k,v])=>`<span>${esc(k)} = ${esc(v)}</span>`).join("")}</div></div>`:"");
  f.rows.forEach(r=>sig[r.name]=r.cells.map(String)); prevSig = sig;
  $("#cap").textContent = f.description;
  $("#pg").style.width = ((fi+1)/n*100)+"%"; $("#ct").textContent = `${fi+1} / ${n}`;
  $("#prev").disabled = fi==0; $("#next").disabled = fi==n-1;
  $("#play").textContent = timer ? "❚❚ Pause" : (fi==n-1 ? "↺ Replay" : "▶ Play");
}
function stop(){ if(timer){clearInterval(timer);timer=null;} }
function step(d){ const n=V().frames.length; fi=Math.min(n-1,Math.max(0,fi+d)); render(); }
function play(){
  if(timer){stop();render();return;}
  if(fi==V().frames.length-1) fi=0;
  render(); timer=setInterval(()=>{ if(fi>=V().frames.length-1){stop();render();return;} step(1); if(fi>=V().frames.length-1){stop();render();} }, 1500*speed);
  render();
}
$("#prev").onclick=()=>{stop();step(-1);}; $("#next").onclick=()=>{stop();step(1);}; $("#play").onclick=play;
$("#spd").onchange=e=>{speed=+e.target.value; if(timer){stop();play();}};
document.addEventListener("keydown",e=>{ if(e.key=="ArrowRight"){stop();step(1);} else if(e.key=="ArrowLeft"){stop();step(-1);} else if(e.key==" "){e.preventDefault();play();} });
document.title = `${P.id}. ${P.title}`;
renderTabs(); renderStatic(); render();
})();
