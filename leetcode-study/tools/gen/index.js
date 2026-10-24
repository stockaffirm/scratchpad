const fs=require("fs"),path=require("path");
const root=path.join(process.env.HOME,"Claude","leetcode-study");
const order=eval("(function(){var window={};"+fs.readFileSync(path.join(root,"assets/order.js"),"utf8")+";return window.ORDER})()");
const src=fs.readFileSync(path.join(__dirname,"build.js"),"utf8");
const g75=eval(src.match(/const groups=(\[\[.*?\]\]);/s)[1]);
const g150=eval(src.match(/const groups150=(\[\[.*?\]\]);/s)[1]);
const extra=fs.existsSync(path.join(__dirname,"sets.json"))?JSON.parse(fs.readFileSync(path.join(__dirname,"sets.json"),"utf8")):[];
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const byId={};for(const d of order){const P=eval("(function(){var window={};"+fs.readFileSync(path.join(root,d,"problem.js"),"utf8")+";return window.PROBLEM})()");byId[P.id]={d,P}}
const in75=new Set(g75.flatMap(g=>g[1])),in150=new Set(g150.flatMap(g=>g[1]));
const member=[{key:"75",title:"LeetCode 75",ids:in75},{key:"150",title:"LeetCode 150",ids:in150},...extra.map(e=>({key:e.key,title:e.title,ids:new Set(e.groups.flatMap(g=>g[1]))}))];
function list(title,note,groups,keep){let sl=0;
 const topics=groups.map(([name,ids])=>{const rows=ids.filter(keep);if(!rows.length)return"";return `<section class="topic open">
<button class="th" aria-expanded="true"><span class="chev">▾</span><span class="tn">${esc(name)}</span></button>
<div class="tb">
${rows.map(id=>{const{d,P}=byId[id];sl++;return `<div class="rowx" data-id="${id}" data-s="${member.filter(m=>m.ids.has(+id)).map(m=>m.key).join(" ")}">
<div class="r"><button class="qt" aria-expanded="false" title="Show question">▸</button><span class="sl">${sl}</span><span class="lc">#${id}</span><a class="ti" href="${d}/">${esc(P.title)}</a><span class="tag ${P.level}">${P.level}</span><button class="ck" role="checkbox" aria-checked="false" title="Mark as done"></button></div>
<div class="qa"><p class="q">${esc(P.question)}</p><div class="io"><div><small>Example input</small>${esc(P.input)}</div><div class="arrow">→</div><div><small>Output</small>${esc(P.output)}</div></div><p class="why"><b>Why?</b> ${esc(P.why)}</p><p class="idea"><b>💡 Key idea</b> ${esc(P.idea)}</p>${P.ex2?`<p class="why edge"><b>Edge / failure case</b> ${esc(P.ex2.input)} → ${esc(P.ex2.output)}<br>${esc(P.ex2.why)}</p>`:""}<a class="open" href="${d}/">Open walkthrough ▶</a></div>
</div>`}).join("\n")}
</div>
</section>`}).join("\n");
 if(!sl)return "";
 return `<section class="plist open"><button class="ph" aria-expanded="true"><span class="chev">▾</span>${title}${note?` <small>${note}</small>`:""}</button><div class="pb">
<div class="thead"><span></span><span class="sl">Sl</span><span class="lc">LC #</span><span class="ti">Problem</span><span>Level</span><span class="dh">Done</span></div>
${topics}
</div></section>`}
const html=`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeetCode Study</title><link rel="stylesheet" href="assets/style.css"></head>
<body><div class="wrap"><h1>LeetCode Study</h1>
<div class="tools"><button id="expAll">Expand all</button><button id="colAll">Collapse all</button><span class="gap"></span>${member.map(m=>`<button class="flt" data-flt="${m.key}" aria-pressed="false">${esc(m.title)}</button>`).join("")}</div>
${list("LeetCode 75","",g75,()=>true)}
${list("LeetCode 150","not in the 75",g150,id=>!in75.has(+id))}
${extra.map((e,i)=>{const seen=new Set([in75,in150,...member.slice(2,2+i).map(m=>m.ids)].flatMap(x=>[...x]));return list(esc(e.title),esc(e.note||"new problems only"),e.groups,id=>!seen.has(+id))}).join("\n")}
</div><script src="assets/progress.js"></script><script src="assets/toc.js"></script></body></html>
`;
fs.writeFileSync(path.join(root,"index.html"),html);console.log("extra sets:",extra.map(e=>e.key).join(",")||"none","|","75 list:",in75.size,"150-only rows:",[...in150].filter(i=>!in75.has(i)).length,"overlap:",[...in150].filter(i=>in75.has(i)).length);
