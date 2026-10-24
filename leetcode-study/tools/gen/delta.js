// Usage: node delta.js 1 2 3 4 ...   (or: node delta.js ids.txt)  -> which LeetCode ids are already built vs NEW
const fs=require("fs"),path=require("path");
const root=path.join(process.env.HOME,"Claude","leetcode-study");
const order=eval("(function(){var window={};"+fs.readFileSync(path.join(root,"assets/order.js"),"utf8")+";return window.ORDER})()");
const have=new Map(order.map(d=>[+d.split("-")[0],d]));
let args=process.argv.slice(2);if(args.length===1&&fs.existsSync(args[0]))args=fs.readFileSync(args[0],"utf8").split(/[\s,]+/);
const ids=[...new Set(args.map(Number).filter(Boolean))];
const old=ids.filter(i=>have.has(i)),neu=ids.filter(i=>!have.has(i));
console.log(`requested ${ids.length}: already built ${old.length}, NEW ${neu.length}`);
console.log("already built:",old.join(",")||"-");
console.log("NEW (need curated build):",neu.join(",")||"-");
