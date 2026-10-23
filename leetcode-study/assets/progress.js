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
  function paint(){ rows.forEach(r => { const d = Progress.has(r.dataset.id); r.classList.toggle("done", d); r.querySelector(".ck").setAttribute("aria-checked", d); }); }
  rows.forEach(r => r.querySelector(".ck").onclick = () => { Progress.set(r.dataset.id, !Progress.has(r.dataset.id)); paint(); });
  paint(); window.addEventListener("pageshow", paint);
})();
