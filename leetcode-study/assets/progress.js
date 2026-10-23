/* Completion tracking, stored in this browser (localStorage). */
window.Progress = {
  key: "lc75.done",
  all(){ try { return JSON.parse(localStorage.getItem(this.key) || "{}"); } catch(e){ return {}; } },
  has(id){ return !!this.all()[id]; },
  set(id, on){ const d = this.all(); if(on) d[id] = d[id] || Date.now(); else delete d[id]; try { localStorage.setItem(this.key, JSON.stringify(d)); } catch(e){} },
  clear(){ try { localStorage.removeItem(this.key); } catch(e){} }
};
(function(){
  const cards = [...document.querySelectorAll(".card")]; if(!cards.length) return;
  const top = document.createElement("div"); top.className = "ptop";
  document.querySelector(".sub").after(top);
  function paint(){
    let done = 0, next = null;
    cards.forEach(c => {
      const id = c.querySelector(".num").textContent.replace("#","");
      const d = Progress.has(id); c.classList.toggle("done", d);
      if(d) done++; else if(!next) next = c;
    });
    document.querySelectorAll(".grp").forEach(g => {
      const cs = [...g.nextElementSibling.querySelectorAll(".card")], n = cs.filter(c=>c.classList.contains("done")).length;
      let s = g.querySelector(".pc"); if(!s){ s = document.createElement("span"); s.className = "pc"; g.appendChild(s); }
      s.textContent = n + "/" + cs.length; s.classList.toggle("all", n === cs.length);
    });
    top.innerHTML = `<div class="pbar"><i style="width:${done/cards.length*100}%"></i></div>
      <div class="prow"><b>${done} of ${cards.length} done</b>
      ${next ? `<a class="cont" href="${next.getAttribute("href")}">${done ? "Continue" : "Start"} → ${next.querySelector("h3").textContent}</a>` : "<span>🎉 All complete!</span>"}
      ${done ? `<button id="reset">Reset</button>` : ""}</div>`;
    const r = top.querySelector("#reset");
    if(r) r.onclick = () => { if(confirm("Clear all progress on this browser?")){ Progress.clear(); paint(); } };
  }
  paint(); window.addEventListener("pageshow", paint);
})();
