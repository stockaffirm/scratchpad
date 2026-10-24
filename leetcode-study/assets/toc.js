/* Landing page: collapsible topics + per-problem question preview. */
(function(){
  const setTopic = (t, open) => { t.classList.toggle("open", open); const b = t.querySelector(".th"); b.setAttribute("aria-expanded", open); b.querySelector(".chev").textContent = open ? "▾" : "▸"; };
  const setQ = (r, open) => { r.classList.toggle("qopen", open); const b = r.querySelector(".qt"); b.setAttribute("aria-expanded", open); b.textContent = open ? "▾" : "▸"; };
  document.querySelectorAll(".topic").forEach(t => t.querySelector(".th").onclick = () => setTopic(t, !t.classList.contains("open")));
  document.querySelectorAll(".rowx").forEach(r => r.querySelector(".qt").onclick = () => setQ(r, !r.classList.contains("qopen")));
  const setList = (l, open) => { l.classList.toggle("open", open); const b = l.querySelector(".ph"); b.setAttribute("aria-expanded", open); b.querySelector(".chev").textContent = open ? "▾" : "▸"; };
  document.querySelectorAll(".plist").forEach(l => l.querySelector(".ph").onclick = () => setList(l, !l.classList.contains("open")));
  document.getElementById("expAll").onclick = () => { document.querySelectorAll(".plist").forEach(l => setList(l, true)); document.querySelectorAll(".topic").forEach(t => setTopic(t, true)); document.querySelectorAll(".rowx").forEach(r => setQ(r, true)); };
  document.getElementById("colAll").onclick = () => { document.querySelectorAll(".plist").forEach(l => setList(l, false)); document.querySelectorAll(".rowx").forEach(r => setQ(r, false)); document.querySelectorAll(".topic").forEach(t => setTopic(t, false)); };
})();
/* Filter buttons: highlight every problem that belongs to LeetCode 75 / LeetCode 150. */
(function(){
  const btns = [...document.querySelectorAll(".flt")], rows = [...document.querySelectorAll(".rowx")];
  btns.forEach(b => b.onclick = () => {
    const on = b.getAttribute("aria-pressed") !== "true";
    btns.forEach(x => x.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", on);
    rows.forEach(r => r.classList.toggle("hl", on && (r.dataset.s || "").split(" ").includes(b.dataset.flt)));
  });
})();
