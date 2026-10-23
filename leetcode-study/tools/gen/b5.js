const {prob,R,M,F,range}=require("./lib");
const E="·";
// 1137
prob({id:1137,slug:"n-th-tribonacci-number",title:"N-th Tribonacci Number",level:"Easy",topic:"DP - 1D",
q:"T0=0, T1=1, T2=1, Tn = Tn-1 + Tn-2 + Tn-3. Find Tn.",i:"n = 4",o:"4",
code:["class Solution:","    def tribonacci(self, n):","        if n == 0: return 0","        a, b, c = 0, 1, 1","        for _ in range(n - 2):","            a, b, c = b, c, a + b + c","        return c"],
time:"O(n)",space:"O(1)",explain:"Keep only the last three values.",best:"Rolling three variables instead of an array.",
notes:["Only the previous three terms matter.","n = 1 and n = 2 both return 1."],
frames:L=>[F("Base","T0=0, T1=1, T2=1.",[L("a, b, c = 0")],[R("T",[0,1,1],[0,1,2])]),
F("Step 1","T3 = 0 + 1 + 1 = 2.",[L("a, b, c = b, c")],[R("T",[0,1,1,2],[3])]),
F("Step 2","T4 = 1 + 1 + 2 = 4.",[L("a, b, c = b, c")],[R("T",[0,1,1,2,4],[4])]),
F("Return","T4 = 4",[L("return c")],[R("answer",[4],[0])])]});
// 746
prob({id:746,slug:"min-cost-climbing-stairs",title:"Min Cost Climbing Stairs",level:"Easy",topic:"DP - 1D",
q:"Pay cost[i] to climb from step i by 1 or 2. Start at step 0 or 1; reach the top. Minimum cost?",i:"[1, 100, 1, 1, 1, 100, 1]",o:"4",
code:["class Solution:","    def minCostClimbingStairs(self, cost):","        a, b = 0, 0","        for i in range(2, len(cost) + 1):","            a, b = b, min(b + cost[i-1], a + cost[i-2])","        return b"],
time:"O(n)",space:"O(1)",explain:"dp[i] = cheapest way to stand on step i.",best:"Two rolling values: cost to reach i-2 and i-1.",
notes:["dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2]).","The top is index n, one past the last step."],
frames:L=>{const c=[1,100,1,1,1,100,1],dp=[0,0],f=[F("Start","Standing on step 0 or 1 is free.",[L("a, b = 0, 0")],[R("cost",c),R("dp",dp,[0,1])])];
 for(let i=2;i<=7;i++){const v=Math.min(dp[i-1]+c[i-1],dp[i-2]+c[i-2]);dp.push(v);f.push(F("dp["+i+"]","min("+dp[i-1]+" + "+c[i-1]+", "+dp[i-2]+" + "+c[i-2]+") = "+v,[L("a, b = b, min")],[R("cost",c,[i-1,i-2]),R("dp",dp,[i])]))}
 f.push(F("Return","dp[7] = 4",[L("return b")],[R("dp",dp,[7])]));return f}});
// 198
prob({id:198,slug:"house-robber",title:"House Robber",level:"Medium",topic:"DP - 1D",
q:"Rob houses for maximum money without robbing two adjacent houses.",i:"[2, 7, 9, 3, 1]",o:"12",
code:["class Solution:","    def rob(self, nums):","        prev = cur = 0","        for x in nums:","            prev, cur = cur, max(cur, prev + x)","        return cur"],
time:"O(n)",space:"O(1)",explain:"Skip this house or rob it plus the best from two back.",best:"Rolling DP: best(i) = max(best(i-1), best(i-2) + x).",
notes:["Each house: take it or leave it.","Only the previous two best values are needed."],
frames:L=>{const a=[2,7,9,3,1];let p=0,c=0;const best=[],f=[F("Start","best so far = 0.",[L("prev = cur")],[R("nums",a),R("best",[E])])];
 a.forEach((x,i)=>{const skip=c,rob=p+x,n=Math.max(skip,rob);p=c;c=n;best.push(c);f.push(F("House "+i+" ("+x+")","max(skip: "+skip+", rob: "+rob+") → "+c,[L("prev, cur = cur")],[R("nums",a,[i]),R("best",best,[i])],{prev:p,cur:c}))});
 f.push(F("Return","Max = 12 (rob 2, 9, 1).",[L("return cur")],[R("nums",a,[0,2,4])]));return f}});
// 790
prob({id:790,slug:"domino-and-tromino-tiling",title:"Domino and Tromino Tiling",level:"Medium",topic:"DP - 1D",
q:"Count ways to tile a 2×n board with dominoes and L-trominoes (mod 10⁹+7).",i:"n = 5",o:"24",
code:["class Solution:","    def numTilings(self, n):","        dp = [1, 1, 2]","        for i in range(3, n + 1):","            dp.append((2 * dp[-1] + dp[-3]) % (10**9 + 7))","        return dp[n]"],
time:"O(n)",space:"O(n)",explain:"dp[n] = 2·dp[n-1] + dp[n-3].",best:"Recurrence dp[n] = 2·dp[n-1] + dp[n-3] (found from the tromino patterns).",
notes:["Trominoes must come in interlocking pairs.","The recurrence comes from counting boundary shapes."],
frames:L=>{const dp=[1,1,2],f=[F("Base","dp[0]=1, dp[1]=1, dp[2]=2.",[L("dp = [1, 1, 2]")],[R("dp",dp,[0,1,2])])];
 for(let i=3;i<=5;i++){dp.push(2*dp[i-1]+dp[i-3]);f.push(F("dp["+i+"]","2×"+dp[i-1]+" + "+dp[i-3]+" = "+dp[i],[L("dp.append")],[R("dp",dp,[i,i-1,i-3])]))}
 f.push(F("Return","24 tilings for n = 5.",[L("return dp[n]")],[R("answer",[24],[0])]));return f}});
// 62
prob({id:62,slug:"unique-paths",title:"Unique Paths",level:"Medium",topic:"DP - Multidimensional",
q:"A robot moves only right or down from top-left to bottom-right. How many paths?",i:"m = 3, n = 3",o:"6",
code:["class Solution:","    def uniquePaths(self, m, n):","        row = [1] * n","        for _ in range(1, m):","            for j in range(1, n):","                row[j] += row[j - 1]","        return row[-1]"],
time:"O(m·n)",space:"O(n)",explain:"Paths to a cell = paths from above + paths from the left.",best:"One rolling row: each cell adds the cell to its left.",
notes:["Top row and left column are all 1.","row[j] (old) is 'from above'; row[j-1] is 'from left'."],
frames:L=>{const g=[[1,1,1],[1,E,E],[1,E,E]];const c=()=>g.map(r=>r.slice());const f=[F("Edges","Only one way along the top row and left column.",[L("row = [1]")],[R("paths",c(),["0,0","0,1","0,2","1,0","2,0"],{},"grid")])];
 g[1][1]=2;g[1][2]=3;f.push(F("Row 2","2 = 1+1,  3 = 2+1.",[L("row[j] +=")],[R("paths",c(),["1,1","1,2"],{},"grid")]));
 g[2][1]=3;g[2][2]=6;f.push(F("Row 3","3 = 2+1,  6 = 3+3.",[L("row[j] +=")],[R("paths",c(),["2,1","2,2"],{},"grid")]));
 f.push(F("Return","6 paths.",[L("return row[-1]")],[R("paths",c(),["2,2"],{},"grid")]));return f}});
// 1143
const gridFrames=(title,a,b,init,fill)=>null;
prob({id:1143,slug:"longest-common-subsequence",title:"Longest Common Subsequence",level:"Medium",topic:"DP - Multidimensional",
q:"Length of the longest subsequence common to both strings.",i:'"abcde", "ace"',o:"3",
code:["class Solution:","    def longestCommonSubsequence(self, a, b):","        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]","        for i in range(1, len(a) + 1):","            for j in range(1, len(b) + 1):","                if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1","                else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])","        return dp[-1][-1]"],
time:"O(n·m)",space:"O(n·m)",explain:"dp[i][j] = LCS of a[:i] and b[:j].",best:"2-D DP: match extends the diagonal, otherwise take the better of up/left.",
notes:["Row/col 0 mean the empty prefix → 0.","Match: diagonal + 1. Else: max(up, left)."],
frames:L=>{const a="abcde",b="ace",dp=Array.from({length:6},()=>Array(4).fill(0));const cp=()=>dp.map((r,i)=>r.map((v,j)=>i===0||j===0?v:(dp[i][j]===null?E:v)));
 const f=[F("Table","Rows = prefixes of 'abcde', columns = prefixes of 'ace'. Edges are 0.",[L("dp = [[0]")],[R("dp",dp.map(r=>r.slice()),[],{},"grid")])];
 for(let i=1;i<=5;i++){const hot=[];for(let j=1;j<=3;j++){if(a[i-1]===b[j-1]){dp[i][j]=dp[i-1][j-1]+1;hot.push(i+","+j)}else dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1])}
  f.push(F("Row '"+a[i-1]+"'",hot.length?"Match at "+hot.length+" cell: diagonal + 1.":"No match: take the larger of up / left.",[L("if a[i-1] == b[j-1]"),L("else: dp[i][j]")],[R("dp",dp.map(r=>r.slice()),hot,{},"grid")]))}
 f.push(F("Return","dp[5][3] = 3 ('ace').",[L("return dp[-1][-1]")],[R("dp",dp.map(r=>r.slice()),["5,3"],{},"grid")]));return f}});
// 714
prob({id:714,slug:"best-time-to-buy-and-sell-stock-with-fee",title:"Best Time to Buy and Sell Stock with Transaction Fee",level:"Medium",topic:"DP - Multidimensional",
q:"Unlimited trades, but pay a fee per sale. Maximum profit?",i:"prices=[1,3,2,8,4,9], fee=2",o:"8",
code:["class Solution:","    def maxProfit(self, prices, fee):","        hold, cash = -prices[0], 0","        for p in prices[1:]:","            hold, cash = max(hold, cash - p), max(cash, hold + p - fee)","        return cash"],
time:"O(n)",space:"O(1)",explain:"Two states: holding a share, or holding cash.",best:"State machine DP: hold vs cash, update both each day.",
notes:["cash = best profit with no share.","hold = best profit while owning a share."],
frames:L=>{const p=[1,3,2,8,4,9],f=[F("Day 0","Buy at 1: hold = -1, cash = 0.",[L("hold, cash =")],[R("prices",p,[0]),R("hold",[-1]),R("cash",[0])])];let h=-1,c=0;const H=[-1],C=[0];
 for(let i=1;i<6;i++){const nh=Math.max(h,c-p[i]),nc=Math.max(c,h+p[i]-2);h=nh;c=nc;H.push(h);C.push(c);f.push(F("Day "+i+" (price "+p[i]+")","hold = "+h+", cash = "+c,[L("hold, cash = max")],[R("prices",p,[i]),R("hold",H,[i]),R("cash",C,[i])]))}
 f.push(F("Return","Best profit = 8.",[L("return cash")],[R("cash",C,[5])]));return f}});
// 72
prob({id:72,slug:"edit-distance",title:"Edit Distance",level:"Medium",topic:"DP - Multidimensional",
q:"Minimum insert / delete / replace operations to turn word1 into word2.",i:'"horse" → "ros"',o:"3",
code:["class Solution:","    def minDistance(self, a, b):","        dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]","        for i in range(1, len(a) + 1):","            for j in range(1, len(b) + 1):","                if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]","                else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])","        return dp[-1][-1]"],
time:"O(n·m)",space:"O(n·m)",explain:"dp[i][j] = edits to turn a[:i] into b[:j].",best:"2-D DP: match is free; else 1 + min(delete, insert, replace).",
notes:["First row/column: i or j insertions/deletions.","up = delete, left = insert, diagonal = replace."],
frames:L=>{const a="horse",b="ros",dp=Array.from({length:6},(_,i)=>Array.from({length:4},(_,j)=>i*j===0?i+j:0));
 const f=[F("Table","Turning '' into 'ros' takes 0–3 inserts; 'horse' into '' takes 0–5 deletes.",[L("dp = [[i + j")],[R("dp",dp.map(r=>r.slice()),[],{},"grid")])];
 for(let i=1;i<=5;i++){const hot=[];for(let j=1;j<=3;j++){if(a[i-1]===b[j-1]){dp[i][j]=dp[i-1][j-1];hot.push(i+","+j)}else dp[i][j]=1+Math.min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])}
 f.push(F("Row '"+a[i-1]+"'",hot.length?"Match → copy the diagonal (free).":"No match: 1 + min of three neighbours.",[L("if a[i-1] == b[j-1]"),L("else: dp[i][j]")],[R("dp",dp.map(r=>r.slice()),hot,{},"grid")]))}
 f.push(F("Return","dp[5][3] = 3 edits.",[L("return dp[-1][-1]")],[R("dp",dp.map(r=>r.slice()),["5,3"],{},"grid")]));return f}});
// 338
prob({id:338,slug:"counting-bits",title:"Counting Bits",level:"Easy",topic:"Bit Manipulation",
q:"For every i in 0…n, count the 1-bits in i's binary form.",i:"n = 5",o:"[0, 1, 1, 2, 1, 2]",
code:["class Solution:","    def countBits(self, n):","        ans = [0] * (n + 1)","        for i in range(1, n + 1):","            ans[i] = ans[i >> 1] + (i & 1)","        return ans"],
time:"O(n)",space:"O(n)",explain:"i has the bits of i/2 plus its last bit.",best:"DP on bits: ans[i] = ans[i >> 1] + (i & 1).",
notes:["i >> 1 drops the last bit.","i & 1 is that last bit."],
frames:L=>{const a=[0],bin=[0,1,10,11,100,101],f=[F("Start","ans[0] = 0.",[L("ans = [0]")],[R("i",[0]),R("ans",[0],[0])])];
 for(let i=1;i<=5;i++){a.push(a[i>>1]+(i&1));f.push(F("i = "+i+" ("+i.toString(2)+")","ans["+(i>>1)+"] + "+(i&1)+" = "+a[i],[L("ans[i] = ans[i >> 1]")],[R("i",range(0,i),[i]),R("ans",a,[i,i>>1])]))}
 f.push(F("Return","[0,1,1,2,1,2]",[L("return ans")],[R("ans",a,range(0,5))]));return f}});
// 136
prob({id:136,slug:"single-number",title:"Single Number",level:"Easy",topic:"Bit Manipulation",
q:"Every number appears twice except one. Find it in O(1) space.",i:"[4, 1, 2, 1, 2]",o:"4",
code:["class Solution:","    def singleNumber(self, nums):","        res = 0","        for x in nums: res ^= x","        return res"],
time:"O(n)",space:"O(1)",explain:"XOR cancels equal pairs.",best:"XOR everything: x ^ x = 0 and 0 ^ y = y.",
notes:["XOR is commutative and associative.","Pairs vanish; the single remains."],
frames:L=>{const a=[4,1,2,1,2],f=[F("Start","res = 0",[L("res = 0")],[R("nums",a),R("res",[0])])];let r=0;
 a.forEach((x,i)=>{const o=r;r^=x;f.push(F("^ "+x,o+" XOR "+x+" = "+r,[L("res ^= x")],[R("nums",a,[i]),R("res",[r],[0])]))});
 f.push(F("Return","Pairs cancelled → 4.",[L("return res")],[R("answer",[r],[0])]));return f}});
// 1318
prob({id:1318,slug:"minimum-flips-to-make-a-or-b-equal-to-c",title:"Minimum Flips to Make a OR b Equal to c",level:"Medium",topic:"Bit Manipulation",
q:"Fewest bit flips in a and/or b so that (a OR b) == c.",i:"a=2 (010), b=6 (110), c=5 (101)",o:"3",
code:["class Solution:","    def minFlips(self, a, b, c):","        flips = 0","        for i in range(32):","            x, y, z = (a >> i) & 1, (b >> i) & 1, (c >> i) & 1","            if z == 0: flips += x + y","            else: flips += 1 if x + y == 0 else 0","        return flips"],
time:"O(1)",space:"O(1)",explain:"Decide bit by bit.",best:"Check each bit: c-bit 0 needs both zero; c-bit 1 needs at least one.",
notes:["c bit 0: flip every 1 in a and b.","c bit 1: flip one only if both are 0."],
frames:L=>[F("Bits","Compare bit by bit, right to left.",[L("def minFlips")],[R("a",[0,1,0]),R("b",[1,1,0]),R("c",[1,0,1])]),
F("Bit 0","a=0, b=0, c=1 → need a 1 → 1 flip.",[L("else: flips")],[R("a",[0,1,0],[2]),R("b",[1,1,0],[2]),R("c",[1,0,1],[2])],{flips:1}),
F("Bit 1","a=1, b=1, c=0 → both must be 0 → 2 flips.",[L("if z == 0")],[R("a",[0,1,0],[1]),R("b",[1,1,0],[1]),R("c",[1,0,1],[1])],{flips:3}),
F("Bit 2","a=0, b=1, c=1 → already OK.",[L("else: flips")],[R("a",[0,1,0],[0]),R("b",[1,1,0],[0]),R("c",[1,0,1],[0])],{flips:3}),
F("Return","3 flips.",[L("return flips")],[R("answer",[3],[0])])]});
// 739
prob({id:739,slug:"daily-temperatures",title:"Daily Temperatures",level:"Medium",topic:"Monotonic Stack",
q:"For each day, how many days until a warmer one? (0 if never.)",i:"[73,74,75,71,69,72,76,73]",o:"[1,1,4,2,1,1,0,0]",
code:["class Solution:","    def dailyTemperatures(self, temps):","        ans, stack = [0] * len(temps), []","        for i, t in enumerate(temps):","            while stack and temps[stack[-1]] < t:","                j = stack.pop(); ans[j] = i - j","            stack.append(i)","        return ans"],
time:"O(n)",space:"O(n)",explain:"Stack of days still waiting for a warmer one.",best:"Monotonic decreasing stack of indices.",
notes:["A warmer day resolves every colder day on the stack.","Each index is pushed and popped once."],
frames:L=>{const T=[73,74,75,71,69,72,76,73],ans=Array(8).fill(0),st=[],f=[F("Input","Stack holds days waiting for a warmer day.",[L("ans, stack")],[R("temps",T),R("ans",ans)])];
 T.forEach((t,i)=>{const r=[];while(st.length&&T[st[st.length-1]]<t){const j=st.pop();ans[j]=i-j;r.push(j)}st.push(i);f.push(F("Day "+i+" ("+t+")",r.length?"Warmer than days "+r.join(", ")+" → resolve them.":"Not warmer than the top; wait.",[L("while stack and"),L("stack.append")],[R("temps",T,[i],M({i})),R("stack (days)",st,[st.length-1]),R("ans",ans,r)]))});
 f.push(F("Return","Days left on the stack never get warmer → 0.",[L("return ans")],[R("ans",ans,range(0,7))]));return f}});
// 901
prob({id:901,slug:"online-stock-span",title:"Online Stock Span",level:"Medium",topic:"Monotonic Stack",
q:"For each price, count consecutive previous days (including today) with price ≤ today's.",i:"100, 80, 60, 70, 60, 75, 85",o:"1, 1, 1, 2, 1, 4, 6",
code:["class StockSpanner:","    def __init__(self):","        self.stack = []","    def next(self, price):","        span = 1","        while self.stack and self.stack[-1][0] <= price:","            span += self.stack.pop()[1]","        self.stack.append((price, span))","        return span"],
time:"O(1) amortised",space:"O(n)",explain:"Merge smaller days into one (price, span) entry.",best:"Monotonic stack of (price, span) pairs.",
notes:["Popped entries are absorbed into the span.","Stack stays strictly decreasing in price."],
frames:L=>{const P=[100,80,60,70,60,75,85],st=[],out=[],f=[F("Start","Stack of (price, span) pairs.",[L("self.stack = []")],[R("prices",P),R("stack",[E])])];
 P.forEach((p,i)=>{let s=1;while(st.length&&st[st.length-1][0]<=p)s+=st.pop()[1];st.push([p,s]);out.push(s);f.push(F("price "+p,"span = "+s,[L("while self.stack"),L("self.stack.append")],[R("prices",P,[i]),R("stack (price,span)",st.map(x=>"("+x+")"),[st.length-1]),R("spans",out,[i])]))});return f}});
// 435
prob({id:435,slug:"non-overlapping-intervals",title:"Non-overlapping Intervals",level:"Medium",topic:"Intervals",
q:"Remove the fewest intervals so the rest don't overlap.",i:"[[1,2],[2,3],[3,4],[1,3]]",o:"1",
code:["class Solution:","    def eraseOverlapIntervals(self, intervals):","        intervals.sort(key=lambda x: x[1])","        end, removed = float('-inf'), 0","        for s, e in intervals:","            if s >= end: end = e","            else: removed += 1","        return removed"],
time:"O(n log n)",space:"O(1)",explain:"Sort by end; keep what finishes earliest.",best:"Greedy: sort by end time and keep any interval that starts after the last kept end.",
notes:["Earliest end leaves the most room.","Touching ends (s == end) are fine."],
frames:L=>{const I=[[1,2],[2,3],[1,3],[3,4]],n=I.map(x=>"["+x+"]"),f=[F("Sort by end","[1,2] [2,3] [1,3] [3,4]",[L("intervals.sort")],[R("intervals",n)])];let end=-Infinity,rm=0;
 I.forEach(([s,e],i)=>{let d,keep=s>=end;if(s>=end){end=e;d="Starts "+s+" ≥ end → keep, end = "+e}else{rm++;d="Starts "+s+" < end "+end+" → overlap, remove"}f.push(F("["+s+","+e+"]",d,[L("for s, e"),L(keep?"if s >= end":"else: removed")],[R("intervals",n,[i]),R("end",[end===-Infinity?"-∞":end],[0])],{removed:rm}))});
 f.push(F("Return","Removed 1 ([1,3]).",[L("return removed")],[R("answer",[rm],[0])]));return f}});
// 452
prob({id:452,slug:"minimum-number-of-arrows-to-burst-balloons",title:"Minimum Number of Arrows to Burst Balloons",level:"Medium",topic:"Intervals",
q:"Balloons span [start, end] on the x-axis. An arrow at x bursts every balloon covering x. Fewest arrows?",i:"[[10,16],[2,8],[1,6],[7,12]]",o:"2",
code:["class Solution:","    def findMinArrowShots(self, points):","        points.sort(key=lambda p: p[1])","        arrows, end = 0, float('-inf')","        for s, e in points:","            if s > end:","                arrows += 1; end = e","        return arrows"],
time:"O(n log n)",space:"O(1)",explain:"Shoot at the earliest end; it bursts everything overlapping it.",best:"Greedy: sort by end, shoot at the end of a new group.",
notes:["Arrow at the earliest end covers the most balloons.","Start > end → a new arrow is needed."],
frames:L=>{const P=[[1,6],[2,8],[7,12],[10,16]],n=P.map(x=>"["+x+"]"),f=[F("Sort by end","[1,6] [2,8] [7,12] [10,16]",[L("points.sort")],[R("balloons",n)])];let end=-Infinity,ar=0;
 P.forEach(([s,e],i)=>{let d,sh=false;if(s>end){ar++;end=e;d="Start "+s+" is past the last arrow → new arrow at "+e;sh=true}else d="Start "+s+" ≤ "+end+" → already burst";f.push(F("["+s+","+e+"]",d,[L("for s, e"),...(sh?[L("arrows += 1")]:[L("if s > end")])],[R("balloons",n,[i]),R("arrow at",[end],[0])],{arrows:ar}))});
 f.push(F("Return","2 arrows (at 6 and 12).",[L("return arrows")],[R("answer",[ar],[0])]));return f}});
// 208
prob({id:208,slug:"implement-trie",title:"Implement Trie (Prefix Tree)",level:"Medium",topic:"Trie",
q:"Build a trie with insert, search (whole word) and startsWith (prefix).",i:"insert(apple), search(apple), search(app), startsWith(app), insert(app), search(app)",o:"T, F, T, T",
code:["class Trie:","    def __init__(self): self.root = {}","    def insert(self, word):","        node = self.root","        for ch in word: node = node.setdefault(ch, {})","        node['$'] = True","    def search(self, word):","        node = self._walk(word)","        return bool(node and '$' in node)","    def startsWith(self, prefix):","        return self._walk(prefix) is not None","    def _walk(self, s):","        node = self.root","        for ch in s:","            if ch not in node: return None","            node = node[ch]","        return node"],
time:"O(len) per op",space:"O(total letters)",explain:"Nested dicts, one level per letter; '$' marks a word end.",best:"Dict-of-dicts trie; '$' separates 'is a prefix' from 'is a word'.",
notes:["search needs the '$' end marker; startsWith doesn't.","Shared prefixes share nodes."],
frames:L=>[F("insert('apple')","One node per letter, then an end marker.",[L("node.setdefault"),L("node['$']")],[R("trie",["a","p","p","l","e","$"],[0,1,2,3,4,5],{},"list")]),
F("search('apple')","Path exists and ends with $ → True.",[L("return bool(node and")],[R("walk",["a","p","p","l","e","$"],[0,1,2,3,4,5],{},"list")],{result:"True"}),
F("search('app')","Path exists, but no $ after the 2nd p → False.",[L("return bool(node and")],[R("walk",["a","p","p"],[0,1,2],{},"list"),R("after p",["l","(no $)"],[1])],{result:"False"}),
F("startsWith('app')","Path exists → True.",[L("return self._walk(prefix)")],[R("walk",["a","p","p"],[0,1,2],{},"list")],{result:"True"}),
F("insert('app')","Reuse a-p-p, add $ after the 2nd p.",[L("node['$'] = True")],[R("trie",["a","p","p","$"],[3],{},"list")]),
F("search('app')","Now $ exists → True.",[L("return bool(node and")],[R("walk",["a","p","p","$"],[0,1,2,3],{},"list")],{result:"True"})]});
// 1268
prob({id:1268,slug:"search-suggestions-system",title:"Search Suggestions System",level:"Medium",topic:"Trie",
q:"After each typed letter, suggest up to 3 products with that prefix, in lexicographic order.",i:'products=[mobile, mouse, moneypot, monitor, mousepad], word="mouse"',o:"[[mobile,moneypot,monitor], same, [mouse,mousepad], same, same]",
code:["class Solution:","    def suggestedProducts(self, products, searchWord):","        products.sort()","        res = []","        for i in range(1, len(searchWord) + 1):","            prefix = searchWord[:i]","            k = bisect_left(products, prefix)","            res.append([w for w in products[k:k+3] if w.startswith(prefix)])","        return res"],
time:"O(n log n + L log n)",space:"O(1) extra",explain:"Sorted list + binary search replaces an explicit trie.",best:"Sort once; bisect to the first match and take up to 3.",
notes:["Sorted order = lexicographic suggestion order.","A trie also works but needs more code."],
frames:L=>{const p=["mobile","moneypot","monitor","mouse","mousepad"],pr=["m","mo","mou","mous","mouse"],res=[["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]];
 const f=[F("Sorted","mobile, moneypot, monitor, mouse, mousepad",[L("products.sort")],[R("products",p)])];
 pr.forEach((x,i)=>{const k=p.findIndex(w=>w>=x);f.push(F("Typed '"+x+"'","First match at index "+k+" → take up to 3 that start with '"+x+"'.",[L("bisect_left"),L("res.append")],[R("products",p,range(k,k+res[i].length-1),M({k})),R("suggest",res[i],range(0,res[i].length-1))]))});
 return f}});
