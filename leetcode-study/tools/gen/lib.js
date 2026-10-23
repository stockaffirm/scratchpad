const probs=[];
const R=(name,cells,hot=[],marks={},kind)=>({name,cells:kind==="grid"?cells.map(r=>r.slice()):Array.from(cells),hot:hot.slice(),marks:{...marks},kind});
const M=o=>{const m={};for(const[k,v]of Object.entries(o)){if(v==null||v<0)continue;m[v]=m[v]?m[v]+","+k:k}return m};
const F=(title,description,lines,rows,facts={})=>({title,description,lines,rows,facts});
const range=(a,b)=>Array.from({length:Math.max(0,b-a+1)},(_,i)=>a+i);
function prob(o){probs.push(o)}
function finish(o){
  const code=o.code,L=(s,n=1)=>{let c=0;for(let i=0;i<code.length;i++)if(code[i].includes(s)&&++c===n)return i+1;throw new Error(o.id+": no line '"+s+"'#"+n)};
  const frames=o.frames(L).map(f=>({title:f.title,description:f.description,codeLines:f.lines,facts:f.facts,
    rows:f.rows.map(r=>({name:r.name,cells:r.kind==="grid"?r.cells.map(x=>x.map(String)):r.cells.map(String),hot:r.hot,marks:r.marks,kind:r.kind}))}));
  return {id:String(o.id),title:o.title,level:o.level,topic:o.topic,question:o.x?o.x[0]:o.q,why:o.x?o.x[1]:undefined,idea:o.x?o.x[2]:undefined,input:o.i,output:o.o,notes:o.notes||[],best:o.best||(o.x&&o.x[2]),
    variants:[{name:o.name||"Solution",code:code.join("\n"),time:o.time,space:o.space,explain:o.explain||(o.x&&o.x[2]),frames}]};
}
module.exports={prob,R,M,F,range,probs,finish};
