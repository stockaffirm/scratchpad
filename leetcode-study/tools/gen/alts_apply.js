const fs=require("fs"),path=require("path");
const root=path.join(process.env.HOME,"Claude","leetcode-study");
const alts={},first={},tags={},names={};
const alt=(id,o)=>{(alts[id]=alts[id]||[]).push(o)};
const best=(id,name)=>{first[id]=name};
const rename=(id,n)=>{names[id]=n};
const tag=(id,name,t)=>{(tags[id]=tags[id]||{})[name]=t};
const files=fs.readdirSync(__dirname).filter(f=>/^alts\d+\.js$/.test(f)).sort();
for(const f of files)require("./"+f)({alt,best,tag,rename});
let n=0,added=0;
for(const d of fs.readdirSync(root)){const p=path.join(root,d,"problem.js");if(!fs.existsSync(p))continue;
 const P=eval("(function(){var window={};"+fs.readFileSync(p,"utf8")+";return window.PROBLEM})()");
 P.variants=P.variants.filter(v=>!v.alt);
 if(names[P.id])P.variants[0].name=names[P.id];
 for(const a of alts[P.id]||[]){P.variants.push({name:a.name,tag:a.tag||"Alternative",code:a.code.join("\n"),time:a.time,space:a.space,explain:a.explain,frames:[],alt:true});added++}
 for(const [nm,t] of Object.entries(tags[P.id]||{})){const v=P.variants.find(x=>x.name===nm);if(v)v.tag=t;else throw new Error(P.id+" no variant "+nm)}
 if(first[P.id]){const i=P.variants.findIndex(v=>v.name===first[P.id]);if(i<0)throw new Error(P.id+" no variant "+first[P.id]);if(i>0)P.variants.unshift(...P.variants.splice(i,1))}
 P.variants[0].tag=P.variants[0].tag&&P.variants[0].tag!=="Alternative"?P.variants[0].tag:"Optimal";
 fs.writeFileSync(p,"window.PROBLEM = "+JSON.stringify(P,null,1)+";\n");n++}
console.log("alts applied:",added,"added to",n,"problems");
