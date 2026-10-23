const fs=require("fs"),path=require("path"),plain=require("./plain");
const root=path.join(process.env.HOME,"Claude","leetcode-study");let n=0;
for(const d of fs.readdirSync(root)){const p=path.join(root,d,"problem.js");if(!fs.existsSync(p))continue;
 const P=eval("(function(){var window={};"+fs.readFileSync(p,"utf8")+";return window.PROBLEM})()");const x=plain[P.id];if(!x){console.log("missing",P.id);continue}
 P.question=x[0];P.why=x[1];P.idea=x[2];fs.writeFileSync(p,"window.PROBLEM = "+JSON.stringify(P,null,1)+";\n");n++}
console.log("updated",n);
