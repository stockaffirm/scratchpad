const fs=require("fs"),path=require("path");const root=path.join(process.env.HOME,"Claude","leetcode-study");const out={};
for(const d of fs.readdirSync(root)){const p=path.join(root,d,"problem.js");if(!fs.existsSync(p))continue;const P=eval("(function(){var window={};"+fs.readFileSync(p,"utf8")+";return window.PROBLEM})()");out[P.id]=P.variants.map(v=>v.code)}
fs.writeFileSync("problems.json",JSON.stringify(out));console.log(Object.keys(out).length)
