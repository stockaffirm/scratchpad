const fs=require("fs"),path=require("path");const {R,M,range}=require("./lib");
const root=path.join(process.env.HOME,"Claude","leetcode-study");const E="·";
const S=(description,codeLines,rows,facts={})=>({description,codeLines,facts,rows});
const G=(n,g,h=[])=>R(n,g,h,{},"grid");
const reg={};const sub=(id,fn)=>{reg[id]=fn};
// ---------- 238 Product of Array Except Self
sub(238,(L)=>{const n=[1,2,3,4],out=[];
 // variant 0
 {const l0=(s,k)=>L(0,s,k);const Lc=[E,E,E,E],Lv=[1,1,2,6],ex=["L[0] = 1 (nothing to the left of index 0)","L[1] = nums[0] × L[0] = 1 × 1 = 1","L[2] = nums[1] × L[1] = 2 × 1 = 2","L[3] = nums[2] × L[2] = 3 × 2 = 6"];
 out.push({v:0,frame:"Build left products",subs:ex.map((d,i)=>{Lc[i]=Lv[i];return S(d,[i?l0("L[i] = nums[i-1]"):l0("L[0] = 1")],[R("nums",n,i?[i-1]:[]),R("L",Lc.slice(),[i])])})});
 const Rc=[E,E,E,E],Rv=[24,12,4,1],ex2=["R[3] = 1 (nothing to the right of index 3)","R[2] = nums[3] × R[3] = 4 × 1 = 4","R[1] = nums[2] × R[2] = 3 × 4 = 12","R[0] = nums[1] × R[1] = 2 × 12 = 24"];
 out.push({v:0,frame:"Build right products",subs:ex2.map((d,k)=>{const i=3-k;Rc[i]=Rv[i];return S(d,[k?l0("R[i] = nums[i+1]"):l0("R[n-1] = 1")],[R("nums",n,k?[i+1]:[]),R("L",Lv,[]),R("R",Rc.slice(),[i])])})});
 const A=[E,E,E,E],ans=[24,12,8,6];
 out.push({v:0,frame:"Multiply per index",subs:[0,1,2,3].map(i=>{A[i]=ans[i];return S("answer["+i+"] = L["+i+"] × R["+i+"] = "+Lv[i]+" × "+[24,12,4,1][i]+" = "+ans[i],[l0("answer[i] = L[i]*R[i]")],[R("L",Lv,[i]),R("R",[24,12,4,1],[i]),R("answer",A.slice(),[i])])})});}
 // variant 1
 {const l1=(s,k)=>L(1,s,k);const a=[E,E,E,E],v=[1,1,2,6],ex=["answer[0] = 1","answer[1] = nums[0] × answer[0] = 1 × 1 = 1","answer[2] = nums[1] × answer[1] = 2 × 1 = 2","answer[3] = nums[2] × answer[2] = 3 × 2 = 6"];
 out.push({v:1,frame:"Forward pass",subs:ex.map((d,i)=>{a[i]=v[i];return S(d,[i?l1("answer[i] = nums[i-1]"):l1("answer[0] = 1")],[R("nums",n,i?[i-1]:[]),R("answer",a.slice(),[i])],{})})});}
 return out});
// ---------- 1431
sub(1431,(L)=>{const c=[2,3,5,1,3],out=[],tot=[5,6,8,4,6];
 const per=(vi,frame,line)=>{const t=[E,E,E,E,E];out.push({v:vi,frame,subs:c.map((x,i)=>{t[i]=tot[i];return S("Kid "+i+": "+x+" + 3 = "+tot[i]+(tot[i]>=5?" ≥ 5 → True":" < 5 → False"),[L(vi,line)],[R("candies",c,[i]),R("with extra",t.slice(),[i])],{max:5})})})};
 per(0,"Check totals","result.append");per(1,"Check each child","result.append");
 const rm=[2,3,5,5,5],cur=[E,E,E,E,E];out.push({v:1,frame:"Scan and update",subs:c.map((x,i)=>{cur[i]=rm[i];return S(i===0?"Start: biggest = 2":x+" vs biggest "+(i?rm[i-1]:2)+" → "+(x>(i?rm[i-1]:2)?"new biggest "+x:"keep "+rm[i]),[i?L(1,"if value > biggest"):L(1,"biggest = candies[0]")],[R("candies",c,[i]),R("running max",cur.slice(),[i])])})});
 return out});
// ---------- 151
sub(151,(L)=>{const out=[];
 const swaps=(arr,pairs,line,nameFn)=>{const a=arr.slice(),subs=[];for(const[i,j]of pairs){[a[i],a[j]]=[a[j],a[i]];subs.push(S("swap positions "+i+" and "+j,[line],[R("a",a.slice(),[i,j],M({i,j}))]))}return subs};
 const s0=[..."the sky is blue"],ln=L(0,"a[i], a[j] = a[j], a[i]");
 out.push({v:0,frame:"Reverse entire sequence",subs:swaps(s0,[[0,14],[1,13],[2,12],[3,11],[4,10],[5,9],[6,8]],ln)});
 const rev=[..."eulb si yks eht"],pairs=[];let words=[[0,3],[5,6],[8,10],[12,14]];const a=rev.slice(),subs=[];
 for(const[s,e]of words){let i=s,j=e;while(i<j){[a[i],a[j]]=[a[j],a[i]];subs.push(S("reverse the word at "+s+"–"+e+": swap "+i+" and "+j,[ln],[R("a",a.slice(),[i,j],M({i,j}))]));i++;j--}}
 out.push({v:0,frame:"Reverse each individual word",subs});
 const dq=[];out.push({v:1,frame:"Append each word at the front",subs:["the","sky","is","blue"].map(w=>{dq.unshift(w);return S("appendleft('"+w+"') → front of the deque",[L(1,"d.appendleft(word)")],[R("deque",dq.slice(),[0])])})});
 return out});
// ---------- 1071 candidate scan
sub(1071,(L)=>[{v:0,frame:"Check k = 6, 5, 4",subs:[[6,3],[5,4],[4,1]].map(([k,r])=>S("k = "+k+": 9 % "+k+" = "+r+" ≠ 0 → can't repeat evenly, skip",[L(0,"if len1 % k or len2 % k")],[R("candidates",[6,5,4],[[6,5,4].indexOf(k)])]))}]);
// ---------- 322 coin change
sub(322,(L)=>{const co=[1,2,5],dp=[0],out=[],ln=L(0,"dp[a] = min");const mk=a=>{let b=Infinity,parts=[];for(const c of co)if(c<=a){const v=dp[a-c]+1;parts.push("coin "+c+" → dp["+(a-c)+"]+1 = "+v);b=Math.min(b,v)}dp.push(b);return S("amount "+a+": "+parts.join(", ")+" → best "+b,[ln],[R("amount",range(0,11),[a]),R("coins needed",dp.concat(Array(11-a).fill(E)),[a])])};
 const s1=[],s2=[],s3=[];for(let a=1;a<=5;a++)s1.push(mk(a));for(let a=6;a<=10;a++)s2.push(mk(a));s3.push(mk(11));
 out.push({v:0,frame:"Small amounts",subs:s1},{v:0,frame:"Up to 10",subs:s2},{v:0,frame:"dp[11]",subs:s3});return out});
// ---------- 1143 LCS
sub(1143,(L)=>{const a="abcde",b="ace",dp=Array.from({length:6},()=>Array(4).fill(0)),out=[];
 for(let i=1;i<=5;i++){const subs=[];for(let j=1;j<=3;j++){let d,ln;if(a[i-1]===b[j-1]){dp[i][j]=dp[i-1][j-1]+1;d="'"+a[i-1]+"' = '"+b[j-1]+"' → diagonal "+dp[i-1][j-1]+" + 1 = "+dp[i][j];ln=L(0,"if a[i-1] == b[j-1]")}else{dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);d="'"+a[i-1]+"' ≠ '"+b[j-1]+"' → max(up "+dp[i-1][j]+", left "+dp[i][j-1]+") = "+dp[i][j];ln=L(0,"else: dp[i][j]")}
  subs.push(S(d,[ln],[G("dp",dp.map(r=>r.slice()),[i+","+j])]))}out.push({v:0,frame:"Row '"+a[i-1]+"'",subs})}return out});
// ---------- 72 edit distance
sub(72,(L)=>{const a="horse",b="ros",dp=Array.from({length:6},(_,i)=>Array.from({length:4},(_,j)=>i*j===0?i+j:0)),out=[];
 for(let i=1;i<=5;i++){const subs=[];for(let j=1;j<=3;j++){let d,ln;if(a[i-1]===b[j-1]){dp[i][j]=dp[i-1][j-1];d="'"+a[i-1]+"' = '"+b[j-1]+"' → copy diagonal "+dp[i][j]+" (free)";ln=L(0,"if a[i-1] == b[j-1]")}else{const u=dp[i-1][j],l=dp[i][j-1],g=dp[i-1][j-1];dp[i][j]=1+Math.min(u,l,g);d="'"+a[i-1]+"' ≠ '"+b[j-1]+"' → 1 + min(up "+u+", left "+l+", diag "+g+") = "+dp[i][j];ln=L(0,"else: dp[i][j]")}
  subs.push(S(d,[ln],[G("dp",dp.map(r=>r.slice()),[i+","+j])]))}out.push({v:0,frame:"Row '"+a[i-1]+"'",subs})}return out});
// ---------- 62 unique paths
sub(62,(L)=>{const g=[[1,1,1],[1,E,E],[1,E,E]],cp=()=>g.map(r=>r.slice()),ln=L(0,"row[j] +=");const out=[],mk=(i,j)=>{const u=g[i-1][j],l=g[i][j-1];g[i][j]=u+l;return S("cell ("+i+","+j+") = above "+u+" + left "+l+" = "+g[i][j],[ln],[G("paths",cp(),[i+","+j])])};
 out.push({v:0,frame:"Row 2",subs:[mk(1,1),mk(1,2)]},{v:0,frame:"Row 3",subs:[mk(2,1),mk(2,2)]});return out});
// ---------- 64 min path sum
sub(64,(L)=>{const gr=[[1,3,1],[1,5,1],[4,2,1]],cp=()=>gr.map(r=>r.slice()),ln=L(0,"grid[i][j] += min(up, left)"),out=[];
 const cell=(i,j)=>{const up=i?gr[i-1][j]:Infinity,left=j?gr[i][j-1]:Infinity,orig=gr[i][j];gr[i][j]=orig+Math.min(up,left);const how=i===0?"only from the left "+left:j===0?"only from above "+up:"min(up "+up+", left "+left+") = "+Math.min(up,left);return S("cell ("+i+","+j+"): "+orig+" + "+how+" = "+gr[i][j],[ln],[G("grid",cp(),[i+","+j])])};
 const r0=[cell(0,1),cell(0,2)],r1=[cell(1,0),cell(1,1),cell(1,2)],r2=[cell(2,0),cell(2,1),cell(2,2)];
 out.push({v:0,frame:"Row 0",subs:r0},{v:0,frame:"Row 1",subs:r1},{v:0,frame:"Row 2",subs:r2});return out});
// ---------- 63 unique paths II
sub(63,(L)=>{const ob=[[0,0,0],[0,1,0],[0,0,0]],p=[[1,E,E],[E,E,E],[E,E,E]],cp=()=>p.map(r=>r.slice()),out=[];
 const cell=(i,j)=>{let d;if(ob[i][j]){p[i][j]=0;d="cell ("+i+","+j+") is an obstacle → 0 paths"}else{const u=i?p[i-1][j]:0,l=j?p[i][j-1]:0;p[i][j]=u+l;d="cell ("+i+","+j+") = above "+u+" + left "+l+" = "+p[i][j]}return S(d,[ob[i][j]?L(0,"if g[i][j] == 1"):L(0,"dp[j] += dp[j - 1]")],[G("paths",cp(),[i+","+j])])};
 out.push({v:0,frame:"Row 0",subs:[cell(0,1),cell(0,2)]},{v:0,frame:"Row 1",subs:[cell(1,0),cell(1,1),cell(1,2)]},{v:0,frame:"Row 2",subs:[cell(2,0),cell(2,1),cell(2,2)]});return out});
// ---------- 221 maximal square
sub(221,(L)=>{const m=[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]],dp=Array.from({length:5},()=>Array(6).fill(0)),out=[];
 for(let i=1;i<=4;i++){const subs=[];for(let j=1;j<=5;j++){let d,ln;if(m[i-1][j-1]){const u=dp[i-1][j],l=dp[i][j-1],g=dp[i-1][j-1];dp[i][j]=1+Math.min(u,l,g);d="cell is 1 → 1 + min(up "+u+", left "+l+", diag "+g+") = "+dp[i][j];ln=L(0,"dp[i][j] = 1 + min")}else{dp[i][j]=0;d="cell is 0 → square size 0";ln=L(0,"if m[i-1][j-1]")}
  subs.push(S(d,[ln],[G("dp",dp.map(r=>r.slice()),[i+","+j])]))}out.push({v:0,frame:"Row "+i,subs})}return out});
// ---------- 120 triangle
sub(120,(L)=>{const tri=[[2],[3,4],[6,5,7],[4,1,8,3]],out=[],dp=tri[3].slice(),ln=L(0,"dp[i] = triangle[row][i] + min");
 for(let r=2;r>=0;r--){const subs=[];const old=dp.slice();for(let i=0;i<tri[r].length;i++){dp[i]=tri[r][i]+Math.min(old[i],old[i+1]);subs.push(S("dp["+i+"] = "+tri[r][i]+" + min("+old[i]+", "+old[i+1]+") = "+dp[i],[ln],[R("row",tri[r],[i]),R("dp",dp.slice(0,tri[r].length),[i])]))}
  out.push({v:0,frame:r===2?"Row [6,5,7]":r===1?"Row [3,4]":"Top",subs})}return out});
// ---------- 135 candy
sub(135,(L)=>{const r=[1,0,2],c=[1,1,1],out=[];const left=[],right=[];
 for(let i=1;i<3;i++){const up=r[i]>r[i-1];if(up)c[i]=c[i-1]+1;left.push(S("child "+i+": rating "+r[i]+(up?" > ":" ≤ ")+r[i-1]+" (left neighbour) → "+(up?"candy = "+c[i-1]+" + 1 = "+c[i]:"stays "+c[i]),[L(0,"if ratings[i] > ratings[i-1]")],[R("ratings",r,[i,i-1]),R("candy",c.slice(),[i])]))}
 for(const i of [1,0]){const up=r[i]>r[i+1];if(up)c[i]=Math.max(c[i],c[i+1]+1);right.push(S("child "+i+": rating "+r[i]+(up?" > ":" ≤ ")+r[i+1]+" (right neighbour) → "+(up?"candy = max(current, "+c[i+1]+" + 1) = "+c[i]:"stays "+c[i]),[L(0,"if ratings[i] > ratings[i+1]")],[R("ratings",r,[i,i+1]),R("candy",c.slice(),[i])]))}
 out.push({v:0,frame:"Left → right",subs:left},{v:0,frame:"Right → left",subs:right});return out});
// ---------- 189 rotate array
sub(189,(L)=>{const out=[],ln=L(0,"nums[a], nums[b] = nums[b], nums[a]"),a=[1,2,3,4,5,6,7];
 const rev=(i,j,frame)=>{const subs=[],tmp=a.slice();let lo=i,hi=j;while(lo<hi){[tmp[lo],tmp[hi]]=[tmp[hi],tmp[lo]];subs.push(S("swap positions "+lo+" and "+hi,[ln],[R("nums",tmp.slice(),[lo,hi],M({a:lo,b:hi}))]));lo++;hi--}for(let k=0;k<tmp.length;k++)a[k]=tmp[k];out.push({v:0,frame,subs})};
 rev(0,6,"Reverse all");rev(0,2,"Reverse first k");rev(3,6,"Reverse the rest");return out});
// ---------- 1207 counter / 242 counts
sub(1207,(L)=>{const arr=[1,2,2,1,1,3],cnt={},subs=[];arr.forEach((x,i)=>{cnt[x]=(cnt[x]||0)+1;subs.push(S("see "+x+" → count of "+x+" is now "+cnt[x],[L(0,"count = Counter")],[R("arr",arr,[i]),R("counts",Object.entries(cnt).map(([k,v])=>k+"→"+v),[Object.keys(cnt).indexOf(String(x))])]))});return [{v:0,frame:"Counter",subs}]});
sub(242,(L)=>{const mk=(w,frame,line)=>{const cnt={},subs=[];[...w].forEach((ch,i)=>{cnt[ch]=(cnt[ch]||0)+1;subs.push(S("'"+ch+"' → count of '"+ch+"' is "+cnt[ch],[L(0,line)],[R("word",w,[i]),R("counts",Object.entries(cnt).map(([k,v])=>k+":"+v),[Object.keys(cnt).indexOf(ch)])]))});return {v:0,frame,subs}};return [mk("anagram","Count s","Counter(s)"),mk("nagaram","Count t","Counter(t)")]});
require('./subs2')({sub,S,G,R,M,range,E});
// ====================== apply ======================
function norm(r){return {name:r.name,cells:r.kind==="grid"?r.cells.map(x=>x.map(String)):r.cells.map(String),hot:r.hot||[],marks:r.marks||{},kind:r.kind}}
function autoReveal(f,prev){ // reveal one changed cell at a time when a row gains >=3 cells at once
  let best=null;
  if(/^(Return|Join|Finish|Done|Final|Last word|split|reversed|Trim|Input|Typed|Quadrants|Leaves of|Best group|Match|Set of|Same|Counts|Take |Start with)/i.test(f.title))return null;
  for(const r of f.rows){if(/^(answer|result|res|answers|normalized|reversed|words|merged|i|ans|right side|left over)$/i.test(r.name))continue;const pr=prev&&prev.rows.find(x=>x.name===r.name);
   if(r.kind==="grid"){ if(!pr||pr.kind!=="grid")continue;const ch=[];r.cells.forEach((row,ri)=>row.forEach((c,ci)=>{if(String(pr.cells[ri]&&pr.cells[ri][ci])!==String(c))ch.push([ri,ci])}));if(ch.length>=3&&(!best||ch.length>best.ch.length))best={r,pr,ch,grid:true}}
   else{ if(r.kind==="list"||(pr&&pr.cells.length!==r.cells.length))continue;const ch=[];r.cells.forEach((c,i)=>{const old=pr?pr.cells[i]:E;if(String(old)!==String(c)&&String(c)!==E)ch.push(i)});if(ch.length>=3&&(!best||ch.length>best.ch.length))best={r,pr,ch}} }
  if(!best)return null;const{r,pr,ch,grid}=best;
  return ch.map((c,k)=>{const rows=f.rows.map(x=>{if(x.name!==r.name)return x;
    if(grid){const cells=x.cells.map((row,ri)=>row.map((v,ci)=>{const done=ch.slice(0,k+1).some(([a,b])=>a===ri&&b===ci);const was=ch.some(([a,b])=>a===ri&&b===ci);return was&&!done?(pr?pr.cells[ri][ci]:E):v}));return {...x,cells,hot:[c[0]+","+c[1]]}}
    const cells=x.cells.map((v,i)=>{const was=ch.includes(i),done=ch.indexOf(i)<=k;return was&&!done?(pr?pr.cells[i]:E):v});return {...x,cells,hot:[c]}});
   const val=grid?r.cells[c[0]][c[1]]:r.cells[c];const where=grid?"["+c[0]+"]["+c[1]+"]":"["+c+"]";
   return {description:r.name+where+" = "+val+(k===ch.length-1?". "+f.description:""),codeLines:f.codeLines,facts:k===ch.length-1?f.facts:{},rows}})}
let manual=0,auto=0,files=0;
for(const d of fs.readdirSync(root)){const p=path.join(root,d,"problem.js");if(!fs.existsSync(p))continue;
 const P=eval("(function(){var window={};"+fs.readFileSync(p,"utf8")+";return window.PROBLEM})()");
 P.variants.forEach(v=>v.frames.forEach(f=>{delete f.sub}));
 const L=(vi,s,n=1)=>{const c=P.variants[vi].code.split("\n");let k=0;for(let i=0;i<c.length;i++)if(c[i].includes(s)&&++k===n)return i+1;throw new Error(P.id+" no line "+s)};
 if(reg[P.id]){for(const o of reg[P.id](L)){const f=P.variants[o.v].frames.find(x=>x.title===o.frame);if(!f)throw new Error(P.id+" no frame "+o.frame);f.sub=o.subs.map(s=>({description:s.description,codeLines:s.codeLines,facts:s.facts,rows:s.rows.map(norm)}));manual++}}
 P.variants.forEach(v=>v.frames.forEach((f,i)=>{if(f.sub||i===0)return;const a=autoReveal(f,v.frames[i-1]);if(a){f.sub=a;auto++}}));
 fs.writeFileSync(p,"window.PROBLEM = "+JSON.stringify(P,null,1)+";\n");files++}
console.log("files",files,"manual frames",manual,"auto frames",auto);
