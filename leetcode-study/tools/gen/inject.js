const fs=require("fs"),path=require("path");const root=path.join(process.env.HOME,"Claude","leetcode-study");
const tr=JSON.parse(fs.readFileSync(path.join(__dirname,"traces.json"),"utf8"));const ex2=JSON.parse(fs.readFileSync(path.join(__dirname,"ex2.json"),"utf8"));let n=0,steps=0;
for(const d of fs.readdirSync(root)){const p=path.join(root,d,"problem.js");if(!fs.existsSync(p))continue;
 const P=eval("(function(){var window={};"+fs.readFileSync(p,"utf8")+";return window.PROBLEM})()");const t=tr[P.id];if(!t)continue;
 P.variants.forEach((v,i)=>{if(t[i]){const {t2,...main}=t[i];v.trace=main;if(t2)v.trace2=t2;steps+=main.steps.length}});if(ex2[P.id])P.ex2=ex2[P.id];
 fs.writeFileSync(p,"window.PROBLEM = "+JSON.stringify(P)+";\n");n++}
console.log("injected traces into",n,"problems;",steps,"steps")
