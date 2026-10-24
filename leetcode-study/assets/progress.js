/* Completion tracking. Always saved in this browser (localStorage); optionally synced to a private
   GitHub repo (progress.json) so it survives cleared storage and follows you across browsers/devices.
   Sync merges per problem by timestamp (latest change wins), so an empty browser can never wipe saved progress. */
(function(){
  const KD = "lc75.done", KU = "lc75.undone", KG = "lc75.gh";
  const REPO = "stockaffirm/leetcode-progress", FILE = "progress.json";
  const ls = {
    get(k, d){ try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch(e){ return d; } },
    set(k, v){ try { localStorage.setItem(k, JSON.stringify(v)); } catch(e){} },
    del(k){ try { localStorage.removeItem(k); } catch(e){} }
  };
  const state = () => ({ done: ls.get(KD, {}), undone: ls.get(KU, {}) });
  const save = s => { ls.set(KD, s.done); ls.set(KU, s.undone); };
  /* latest timestamp wins per problem; a tie keeps it done */
  function merge(a, b){
    const out = { done: {}, undone: {} };
    const ids = new Set([...Object.keys(a.done), ...Object.keys(a.undone), ...Object.keys(b.done), ...Object.keys(b.undone)]);
    ids.forEach(id => {
      const d = Math.max(a.done[id] || 0, b.done[id] || 0), u = Math.max(a.undone[id] || 0, b.undone[id] || 0);
      if(d && d >= u) out.done[id] = d; else if(u) out.undone[id] = u;
    });
    return out;
  }
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  const gh = { cfg: () => ls.get(KG, null), status: "off", listeners: [] };
  const say = (st, msg) => { gh.status = st; gh.msg = msg || ""; gh.listeners.forEach(f => f()); };
  const b64e = s => btoa(unescape(encodeURIComponent(s))), b64d = s => decodeURIComponent(escape(atob(s.replace(/\s/g, ""))));
  async function api(method, body){
    const c = gh.cfg(); if(!c) throw new Error("no token");
    const r = await fetch("https://api.github.com/repos/" + (c.repo || REPO) + "/contents/" + FILE, {
      method, headers: { Authorization: "Bearer " + c.token, Accept: "application/vnd.github+json", "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined, cache: "no-store" });
    if(r.status === 404 && method === "GET") return { sha: null, data: { done: {}, undone: {} } };
    if(!r.ok) { const e = new Error("GitHub " + r.status); e.status = r.status; throw e; }
    const j = await r.json();
    if(method === "GET"){ let data = {}; try { data = JSON.parse(b64d(j.content)); } catch(e){} return { sha: j.sha, data: { done: data.done || {}, undone: data.undone || {} } }; }
    return j;
  }
  let busy = false, again = false, timer = null;
  /* pull -> merge -> save locally -> push if the merge differs from what is on GitHub */
  async function sync(){
    if(!gh.cfg()) return say("off");
    if(busy){ again = true; return; }
    busy = true; say("syncing");
    try {
      for(let tries = 0; tries < 3; tries++){
        const remote = await api("GET");
        const merged = merge(state(), remote.data);
        save(merged); P.paint();
        if(same(merged, remote.data)) { say("ok", "Synced " + new Date().toLocaleTimeString()); break; }
        try {
          await api("PUT", { message: "progress " + Object.keys(merged.done).length + " done", content: b64e(JSON.stringify(merged)), sha: remote.sha || undefined });
          say("ok", "Synced " + new Date().toLocaleTimeString()); break;
        } catch(e){ if(e.status !== 409 && e.status !== 422 || tries === 2) throw e; }
      }
    } catch(e){
      say("err", e.status === 401 || e.status === 403 ? "Token rejected (needs Contents read+write on " + REPO + ")" : e.status === 404 ? "Repo " + REPO + " not found for this token" : "Sync failed: " + e.message);
    }
    busy = false; if(again){ again = false; sync(); }
  }
  const later = () => { clearTimeout(timer); timer = setTimeout(sync, 1200); };

  const P = window.Progress = {
    key: KD, gh,
    all(){ return state().done; },
    has(id){ return !!state().done[id]; },
    set(id, on){
      const s = state(), now = Date.now();
      if(on){ if(s.done[id]) return; s.done[id] = now; delete s.undone[id]; } else { delete s.done[id]; s.undone[id] = now; }
      save(s); later();
    },
    clear(){ ls.del(KD); ls.del(KU); },
    paint(){},
    sync,
    connect(){
      const t = prompt("Paste a GitHub fine-grained token with Contents: read & write on " + REPO + " (kept only in this browser):");
      if(!t) return; ls.set(KG, { token: t.trim(), repo: REPO }); sync();
    },
    disconnect(){ ls.del(KG); say("off"); }
  };
  window.addEventListener("pageshow", () => { if(gh.cfg()) sync(); });
  document.addEventListener("visibilitychange", () => { if(!document.hidden && gh.cfg()) sync(); });

  /* ---- index page: checkboxes + sync button ---- */
  const rows = [...document.querySelectorAll(".rowx")]; if(!rows.length){ if(gh.cfg()) sync(); return; }
  P.paint = () => rows.forEach(r => { const d = P.has(r.dataset.id); r.classList.toggle("done", d); r.querySelector(".ck").setAttribute("aria-checked", d); });
  rows.forEach(r => r.querySelector(".ck").onclick = () => { P.set(r.dataset.id, !P.has(r.dataset.id)); P.paint(); });
  const tools = document.querySelector(".tools");
  if(tools){
    const b = document.createElement("button"); b.id = "syncBtn"; b.className = "sync";
    tools.insertBefore(b, tools.querySelector(".gap"));
    const draw = () => {
      b.textContent = { off: "☁ Sync progress to GitHub", syncing: "☁ Syncing…", ok: "☁ Synced", err: "☁ " + (gh.msg || "Sync problem") }[gh.status];
      b.title = gh.msg || (gh.status === "off" ? "Save done marks in a private GitHub repo" : "Click to " + (gh.status === "ok" ? "disconnect" : "retry"));
      b.dataset.st = gh.status;
    };
    gh.listeners.push(draw);
    b.onclick = () => { if(!gh.cfg()) P.connect(); else if(gh.status === "ok" && confirm("Disconnect GitHub sync in this browser? Your progress stays saved here and on GitHub.")) P.disconnect(); else if(gh.status === "err") sync(); };
    draw();
  }
  P.paint(); window.addEventListener("pageshow", P.paint);
  if(gh.cfg()) sync();
})();
