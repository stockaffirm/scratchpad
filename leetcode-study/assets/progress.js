/* Completion tracking. Always saved in this browser (localStorage); after "Sign in with GitHub" it is also synced
   to Supabase (one private row per GitHub user) so it survives cleared storage and follows you across devices.
   Sync merges per problem by timestamp (latest change wins), so an empty browser can never wipe saved progress. */
(function(){
  const KD = "lc75.done", KU = "lc75.undone";
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

  /* ---- Supabase backend: "Sign in with GitHub", one row per user (RLS) in public.leetcode_progress ---- */
  const SB_URL = "https://mxdljkwconnaanpgvigz.supabase.co", SB_KEY = "sb_publishable_o6BVx6t3iKn1VP4hkrVWQQ_xjofy47K";
  const SB_STORE = "sb-mxdljkwconnaanpgvigz-auth-token";
  const INDEX_URL = new URL("../", (document.currentScript || {}).src || location.href).href;
  const hasSession = () => { try { return !!localStorage.getItem(SB_STORE); } catch(e){ return false; } };
  const wantsSb = () => hasSession() || /[#?&](access_token|code)=/.test(location.hash + location.search);
  const gh = { cfg: () => (wantsSb() || gh.user) ? { user: gh.user } : null, status: "off", listeners: [], user: null };
  const say = (st, msg) => { gh.status = st; gh.msg = msg || ""; gh.listeners.forEach(f => f()); };
  let sbp = null;
  function client(){
    return sbp = sbp || new Promise((res, rej) => {
      const go = () => res(window.supabase.createClient(SB_URL, SB_KEY, { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true } }));
      if(window.supabase) return go();
      const t = document.createElement("script"); t.src = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.45.4/dist/umd/supabase.js";
      t.onload = go; t.onerror = () => { sbp = null; rej(new Error("could not load Supabase")); }; document.head.appendChild(t);
    });
  }
  async function api(method, body){
    const sb = await client(), { data: { session } } = await sb.auth.getSession();
    if(!session){ const e = new Error("signed out"); e.status = 401; throw e; }
    gh.user = session.user.user_metadata.user_name || session.user.email || "you";
    if(method === "GET"){
      const { data, error } = await sb.from("leetcode_progress").select("data").maybeSingle();
      if(error) throw Object.assign(new Error(error.message), { status: 500 });
      const d = (data && data.data) || {}; return { data: { done: d.done || {}, undone: d.undone || {} } };
    }
    const { error } = await sb.from("leetcode_progress").upsert({ user_id: session.user.id, data: body, updated_at: new Date().toISOString() });
    if(error) throw Object.assign(new Error(error.message), { status: 500 });
  }
  let busy = false, again = false, timer = null;
  /* pull -> merge -> save locally -> push if the merge differs from what is stored */
  async function sync(){
    if(!gh.cfg()) return say("off");
    if(busy){ again = true; return; }
    busy = true; say("syncing");
    try {
      const remote = await api("GET");
      const merged = merge(state(), remote.data);
      save(merged); P.paint();
      if(!same(merged, remote.data)) await api("PUT", merged);
      say("ok", "Synced as " + gh.user + " at " + new Date().toLocaleTimeString());
    } catch(e){
      if(e.status === 401){ gh.user = null; say("off"); } else say("err", "Sync failed: " + e.message);
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
    async connect(){ try { const sb = await client(); await sb.auth.signInWithOAuth({ provider: "github", options: { redirectTo: INDEX_URL } }); } catch(e){ say("err", e.message); } },
    async disconnect(){ try { const sb = await client(); await sb.auth.signOut(); } catch(e){} gh.user = null; say("off"); }
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
      b.textContent = { off: "☁ Sign in with GitHub to sync", syncing: "☁ Syncing…", ok: "☁ Synced", err: "☁ " + (gh.msg || "Sync problem") }[gh.status];
      b.title = gh.msg || (gh.status === "off" ? "Save your done marks to your account so they follow you across browsers" : "Click to " + (gh.status === "ok" ? "disconnect" : "retry"));
      b.dataset.st = gh.status;
    };
    gh.listeners.push(draw);
    b.onclick = () => { if(!gh.cfg()) P.connect(); else if(gh.status === "ok" && confirm("Sign out of sync in this browser? Your progress stays saved here and in your account.")) P.disconnect(); else if(gh.status === "err") sync(); };
    draw();
  }
  P.paint(); window.addEventListener("pageshow", P.paint);
  if(gh.cfg()) sync();
})();
