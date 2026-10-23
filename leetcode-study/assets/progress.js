/* Completion tracking, stored in this browser (localStorage). */
window.Progress = {
  key: "lc75.done",
  all(){ try { return JSON.parse(localStorage.getItem(this.key) || "{}"); } catch(e){ return {}; } },
  has(id){ return !!this.all()[id]; },
  set(id, on){ const d = this.all(); if(on) d[id] = d[id] || Date.now(); else delete d[id]; try { localStorage.setItem(this.key, JSON.stringify(d)); } catch(e){} },
  clear(){ try { localStorage.removeItem(this.key); } catch(e){} }
};
(function(){
  const rows = [...document.querySelectorAll(".rowx")]; if(!rows.length) return;
  const top = document.createElement("div"); top.className = "ptop";
  document.querySelector(".sub").after(top);
  function paint(){
    let done = 0, next = null;
    rows.forEach(r => { const d = Progress.has(r.dataset.id); r.classList.toggle("done", d); r.querySelector(".ck").setAttribute("aria-checked", d); if(d) done++; else if(!next) next = r; });
    document.querySelectorAll(".topic").forEach(t => {
      const rs = [...t.querySelectorAll(".rowx")], n = rs.filter(r=>r.classList.contains("done")).length, s = t.querySelector(".pc");
      s.textContent = n + "/" + rs.length; s.classList.toggle("all", n === rs.length);
    });
    top.innerHTML = `<div class="pbar"><i style="width:${done/rows.length*100}%"></i></div>
      <div class="prow"><b>${done} of ${rows.length} done</b>
      ${next ? `<a class="cont" href="${next.querySelector(".ti").getAttribute("href")}">${done ? "Continue" : "Start"} → ${next.querySelector(".ti").textContent}</a>` : "<span>🎉 All complete!</span>"}
      ${done ? `<button id="reset">Reset</button>` : ""}</div>`;
    const r = top.querySelector("#reset");
    if(r) r.onclick = () => { if(confirm("Clear all progress on this browser?")){ Progress.clear(); paint(); } };
  }
  rows.forEach(r => r.querySelector(".ck").onclick = () => { Progress.set(r.dataset.id, !Progress.has(r.dataset.id)); paint(); });
  paint(); window.addEventListener("pageshow", paint);
})();
