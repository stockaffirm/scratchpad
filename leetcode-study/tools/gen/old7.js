// regenerate the 7 original problems from lessons.js so the pipeline is repeatable
const fs=require("fs"),path=require("path"),vm=require("vm");
const root=path.join(process.env.HOME,"Claude","leetcode-study");
const ctx={window:{}};vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(__dirname,"..","lessons.js"),"utf8"),ctx);
const meta={"1071":{slug:"gcd-of-strings",level:"Easy",q:"x",i:'"ABCABCABC", "ABCABC"',o:'"ABC"'},"1431":{slug:"kids-with-candies",level:"Easy",i:"candies=[2,3,5,1,3], extra=3",o:"[T, T, T, F, T]"},"605":{slug:"can-place-flowers",level:"Easy",i:"flowerbed=[1,0,0,0,1], n=1",o:"True"},"345":{slug:"reverse-vowels",level:"Easy",i:'"IceCreAm"',o:'"AceCreIm"'},"151":{slug:"reverse-words",level:"Medium",i:'"  the  sky is blue  "',o:'"blue is sky the"'},"238":{slug:"product-except-self",level:"Medium",i:"[1, 2, 3, 4]",o:"[24, 12, 8, 6]"},"334":{slug:"increasing-triplet",level:"Medium",i:"[2, 1, 5, 0, 4, 6]",o:"True"}};
for(const l of ctx.window.STUDY_LESSONS){const m=meta[l.id],dir=path.join(root,l.id+"-"+m.slug);fs.mkdirSync(dir,{recursive:true});
 const P={id:l.id,title:l.title,level:m.level,topic:l.topic,question:"x",input:m.i,output:m.o,notes:l.notes,best:l.recommendation||l.variants[0].explain,
  variants:l.variants.map(v=>({name:v.name,code:v.code,time:v.time,space:v.space,explain:v.explain,frames:v.frames.map(f=>({title:f.title,description:f.description,codeLines:f.codeLines,facts:f.facts||{},rows:f.rows.map(r=>({name:r.name,cells:r.cells.map(String),hot:r.hot,marks:{}}))}))}))};
 fs.writeFileSync(path.join(dir,"problem.js"),"window.PROBLEM = "+JSON.stringify(P,null,1)+";\n")}
console.log("old7 regenerated");
