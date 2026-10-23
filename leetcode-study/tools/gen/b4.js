const {prob,R,M,F,range}=require("./lib");
const E="·",sorted=a=>a.slice().sort((x,y)=>x-y);
// 841
prob({id:841,slug:"keys-and-rooms",title:"Keys and Rooms",level:"Medium",topic:"Graphs - DFS",
q:"Room 0 is open. Each room holds keys to other rooms. Can you visit them all?",i:"[[1,3],[3,0,1],[2],[0]]",o:"False",
code:["class Solution:","    def canVisitAllRooms(self, rooms):","        seen, stack = {0}, [0]","        while stack:","            r = stack.pop()","            for k in rooms[r]:","                if k not in seen:","                    seen.add(k); stack.append(k)","        return len(seen) == len(rooms)"],
time:"O(n+keys)",space:"O(n)",explain:"DFS from room 0 over keys.",best:"Plain DFS with a visited set; compare the count at the end.",
notes:["Rooms are nodes, keys are directed edges.","Unreached room → False."],
frames:L=>{const rooms=[[1,3],[3,0,1],[2],[0]],seen=[0],st=[0],keys=rooms.map((k,i)=>i+":"+k.join("")),f=[F("Rooms","Keys per room.",[L("def can")],[R("room: keys",keys)])];
 f.push(F("Start","Open room 0.",[L("seen, stack")],[R("room: keys",keys,[0]),R("visited",seen),R("stack",st)]));
 while(st.length){const r=st.pop();const nw=[];for(const k of rooms[r])if(!seen.includes(k)){seen.push(k);st.push(k);nw.push(k)}
  f.push(F("Enter room "+r,nw.length?"New keys → rooms "+nw.join(", ")+".":"No new rooms.",[L("r = stack.pop"),L("seen.add")],[R("room: keys",keys,[r]),R("visited",seen,range(seen.length-nw.length,seen.length-1)),R("stack",st.length?st:[E])]))}
 f.push(F("Return","Visited 3 of 4 rooms — room 2 is never reachable.",[L("return len")],[R("visited",[0,1,2,3].map(i=>seen.includes(i)?i:"✗"),[2])],{result:"False"}));return f}});
// 547
prob({id:547,slug:"number-of-provinces",title:"Number of Provinces",level:"Medium",topic:"Graphs - DFS",
q:"Cities are connected directly or indirectly. Count the groups.",i:"[[1,1,0],[1,1,0],[0,0,1]]",o:"2",
code:["class Solution:","    def findCircleNum(self, isConnected):","        n, seen, count = len(isConnected), set(), 0","        def dfs(i):","            seen.add(i)","            for j in range(n):","                if isConnected[i][j] and j not in seen: dfs(j)","        for i in range(n):","            if i not in seen:","                dfs(i); count += 1","        return count"],
time:"O(n²)",space:"O(n)",explain:"Each unvisited city starts a new province.",best:"DFS flood-fill; count how many times you start one.",
notes:["Provinces = connected components.","Matrix is symmetric."],
frames:L=>{const g=[[1,1,0],[1,1,0],[0,0,1]];return [F("Matrix","1 means directly connected.",[L("def find")],[R("isConnected",g,[],{},"grid")]),
F("Start at city 0","Not seen → province 1. DFS reaches 1 too.",[L("if i not in seen"),L("dfs(i)")],[R("isConnected",g,["0,0","0,1"],{},"grid"),R("seen",[0,1],[0,1])],{count:1}),
F("City 1","Already seen → skip.",[L("if i not in seen")],[R("seen",[0,1],[1])],{count:1}),
F("City 2","Not seen → province 2.",[L("dfs(i)")],[R("isConnected",g,["2,2"],{},"grid"),R("seen",[0,1,2],[2])],{count:2}),
F("Return","2 provinces: {0,1} and {2}.",[L("return count")],[R("provinces",["{0,1}","{2}"],[0,1])])]}});
// 1466
prob({id:1466,slug:"reorder-routes",title:"Reorder Routes to Make All Paths Lead to the City Zero",level:"Medium",topic:"Graphs - DFS",
q:"Roads are one-way. Flip the fewest so every city can reach city 0.",i:"n=6, [[0,1],[1,3],[2,3],[4,0],[4,5]]",o:"3",
code:["class Solution:","    def minReorder(self, n, connections):","        adj = defaultdict(list)","        for a, b in connections:","            adj[a].append((b, 1)); adj[b].append((a, 0))","        seen, changes, stack = {0}, 0, [0]","        while stack:","            city = stack.pop()","            for nxt, cost in adj[city]:","                if nxt not in seen:","                    seen.add(nxt); changes += cost; stack.append(nxt)","        return changes"],
time:"O(n)",space:"O(n)",explain:"Walk away from 0; an edge pointing away costs 1 to flip.",best:"Treat it as an undirected tree; count edges that point away from 0.",
notes:["n cities, n-1 roads → a tree.","Edge a→b seen from a pointing outward must flip."],
frames:L=>{const adj={0:[[1,1],[4,0]],1:[[0,0],[3,1]],2:[[3,1]],3:[[1,0],[2,0]],4:[[0,1],[5,1]],5:[[4,0]]},seen=[0],st=[0];let ch=0;
 const f=[F("Roads","0→1, 1→3, 2→3, 4→0, 4→5",[L("def minReorder")],[R("roads",["0→1","1→3","2→3","4→0","4→5"])])];
 while(st.length){const c=st.pop(),notes=[];for(const[n,w]of adj[c])if(!seen.includes(n)){seen.push(n);ch+=w;st.push(n);notes.push(c+"→"+n+(w?" points away: flip":" already toward "+c))}
  f.push(F("City "+c,notes.length?notes.join("; "):"Nothing new.",[L("city = stack.pop"),L("changes += cost")],[R("visited",seen,[seen.length-1]),R("stack",st.length?st:[E])],{changes:ch}))}
 f.push(F("Return","3 flips: 0→1, 1→3, 4→5.",[L("return changes")],[R("flips",["0→1","1→3","4→5"],[0,1,2])],{changes:3}));return f}});
// 399
prob({id:399,slug:"evaluate-division",title:"Evaluate Division",level:"Medium",topic:"Graphs - DFS",
q:"Given a/b = 2 and b/c = 3, answer queries like a/c and c/a.",i:"a/b=2, b/c=3 ; queries a/c, c/a",o:"[6.0, 0.1667]",
code:["class Solution:","    def calcEquation(self, equations, values, queries):","        g = defaultdict(dict)","        for (a, b), v in zip(equations, values):","            g[a][b] = v; g[b][a] = 1 / v","        def dfs(x, y, seen):","            if x not in g or y not in g: return -1.0","            if x == y: return 1.0","            seen.add(x)","            for nb, w in g[x].items():","                if nb not in seen:","                    r = dfs(nb, y, seen)","                    if r != -1.0: return w * r","            return -1.0","        return [dfs(a, b, set()) for a, b in queries]"],
time:"O(Q·(V+E))",space:"O(V+E)",explain:"Edge weights multiply along a path.",best:"Graph with weight v and 1/v; DFS multiplies weights along the path.",
notes:["a/b = 2 means edge a→b weight 2, b→a weight 1/2.","Unknown variable → -1."],
frames:L=>[F("Build graph","a —2→ b, b —3→ c, plus reverse edges 1/2 and 1/3.",[L("g[a][b] = v")],[R("edges",["a→b 2","b→c 3","b→a ½","c→b ⅓"])]),
F("Query a/c","Path a → b → c.",[L("for nb, w")],[R("path",["a","b","c"],[0,1,2],{},"list")],{}),
F("Multiply","2 × 3 = 6.",[L("return w * r")],[R("weights",[2,3],[0,1]),R("a/c",[6],[0])]),
F("Query c/a","Path c → b → a: ⅓ × ½ = ⅙.",[L("return w * r")],[R("path",["c","b","a"],[0,1,2],{},"list"),R("c/a",["0.1667"],[0])]),
F("Return","[6.0, 0.1667]",[L("return [dfs")],[R("answers",["6.0","0.1667"],[0,1])])]});
// 1926
prob({id:1926,slug:"nearest-exit-from-entrance-in-maze",title:"Nearest Exit from Entrance in Maze",level:"Medium",topic:"Graphs - BFS",
q:"Fewest steps from the entrance to any border exit ('.' cells on the edge).",i:"4×4 maze, entrance (1,0)",o:"4",
code:["class Solution:","    def nearestExit(self, maze, entrance):","        m, n = len(maze), len(maze[0])","        q = deque([(entrance[0], entrance[1], 0)])","        maze[entrance[0]][entrance[1]] = '+'","        while q:","            r, c, d = q.popleft()","            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):","                nr, nc = r + dr, c + dc","                if 0 <= nr < m and 0 <= nc < n and maze[nr][nc] == '.':","                    if nr in (0, m-1) or nc in (0, n-1): return d + 1","                    maze[nr][nc] = '+'","                    q.append((nr, nc, d + 1))","        return -1"],
time:"O(m·n)",space:"O(m·n)",explain:"BFS expands one step at a time, so the first exit found is nearest.",best:"BFS: the first border cell reached is the shortest path.",
notes:["The entrance itself doesn't count as an exit.","Mark cells visited by turning them into walls."],
frames:L=>{const base=[["+","+","+","+"],[".",".",".","+"],["+","+",".","+"],["+","+",".","+"]];const g=(ex)=>base.map((r,i)=>r.map((c,j)=>ex[i+","+j]??c));
 const st=[["S"],["1"],["2"],["3"]];const sim=[{"1,0":"S"},{"1,0":"S","1,1":1},{"1,0":"S","1,1":1,"1,2":2},{"1,0":"S","1,1":1,"1,2":2,"2,2":3},{"1,0":"S","1,1":1,"1,2":2,"2,2":3,"3,2":"E"}];
 return [F("Maze","Start S at (1,0). '+' is a wall; the border '.' cell is the exit.",[L("def nearest")],[R("maze",g(sim[0]),["1,0"],{},"grid")]),
F("Step 1","Expand to (1,1). Not on the border.",[L("r, c, d = q.popleft"),L("q.append")],[R("maze",g(sim[1]),["1,1"],{},"grid")],{d:1}),
F("Step 2","Expand to (1,2).",[L("q.append")],[R("maze",g(sim[2]),["1,2"],{},"grid")],{d:2}),
F("Step 3","Expand down to (2,2).",[L("q.append")],[R("maze",g(sim[3]),["2,2"],{},"grid")],{d:3}),
F("Exit found","(3,2) is on the border → d + 1 = 4.",[L("return d + 1")],[R("maze",g(sim[4]),["3,2"],{},"grid")],{answer:4})]}});
// 994
prob({id:994,slug:"rotting-oranges",title:"Rotting Oranges",level:"Medium",topic:"Graphs - BFS",
q:"Each minute rotten oranges (2) rot adjacent fresh ones (1). Minutes until none are fresh?",i:"[[2,1,1],[1,1,0],[0,1,1]]",o:"4",
code:["class Solution:","    def orangesRotting(self, grid):","        q, fresh = deque(), 0","        for r, row in enumerate(grid):","            for c, v in enumerate(row):","                if v == 2: q.append((r, c))","                elif v == 1: fresh += 1","        minutes = 0","        while q and fresh:","            for _ in range(len(q)):","                r, c = q.popleft()","                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):","                    nr, nc = r + dr, c + dc","                    if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 1:","                        grid[nr][nc] = 2; fresh -= 1; q.append((nr, nc))","            minutes += 1","        return minutes if fresh == 0 else -1"],
time:"O(m·n)",space:"O(m·n)",explain:"Multi-source BFS, one minute per level.",best:"Multi-source BFS: all rotten oranges spread together each minute.",
notes:["Start BFS from every rotten orange at once.","Leftover fresh → -1."],
frames:L=>{let g=[[2,1,1],[1,1,0],[0,1,1]],q=[[0,0]],fresh=6,min=0;const cp=()=>g.map(r=>r.slice());const f=[F("Start","One rotten orange; 6 fresh.",[L("q, fresh"),L("fresh += 1")],[R("grid",cp(),["0,0"],{},"grid")],{fresh,minutes:0})];
 while(q.length&&fresh){const nq=[],hot=[];for(const[r,c]of q)for(const[dr,dc]of[[1,0],[-1,0],[0,1],[0,-1]]){const a=r+dr,b=c+dc;if(a>=0&&a<3&&b>=0&&b<3&&g[a][b]===1){g[a][b]=2;fresh--;nq.push([a,b]);hot.push(a+","+b)}}
  q=nq;min++;f.push(F("Minute "+min,"Rot spreads to "+hot.length+" orange"+(hot.length>1?"s":"")+".",[L("minutes += 1"),L("grid[nr][nc] = 2")],[R("grid",cp(),hot,{},"grid")],{fresh,minutes:min}))}
 f.push(F("Return","No fresh left after "+min+" minutes.",[L("return minutes")],[R("answer",[min],[0])]));return f}});
// 215
prob({id:215,slug:"kth-largest-element",title:"Kth Largest Element in an Array",level:"Medium",topic:"Heap / Priority Queue",
q:"Find the kth largest value (not kth distinct).",i:"[3, 2, 1, 5, 6, 4], k = 2",o:"5",
code:["class Solution:","    def findKthLargest(self, nums, k):","        heap = []","        for x in nums:","            heappush(heap, x)","            if len(heap) > k: heappop(heap)","        return heap[0]"],
time:"O(n log k)",space:"O(k)",explain:"Min-heap of the k largest; the root is the kth largest.",best:"Keep a min-heap of size k; its smallest is the answer.",
notes:["Pop the smallest whenever size exceeds k.","Python's heapq is a min-heap."],
frames:L=>{const a=[3,2,1,5,6,4],k=2,h=[],f=[F("Input","Keep only the 2 largest seen so far.",[L("heap = []")],[R("nums",a),R("heap",[E])])];
 a.forEach((x,i)=>{h.push(x);h.sort((p,q)=>p-q);let d="Push "+x+".";if(h.length>k){const p=h.shift();d+=" Too big → pop smallest ("+p+")."}f.push(F("x = "+x,d,[L("heappush"),L("if len(heap)")],[R("nums",a,[i],M({x:i})),R("heap (min first)",h,[0])]))});
 f.push(F("Return","heap[0] = 5 is the 2nd largest.",[L("return heap[0]")],[R("heap",h,[0])]));return f}});
// 2336
prob({id:2336,slug:"smallest-number-in-infinite-set",title:"Smallest Number in Infinite Set",level:"Medium",topic:"Heap / Priority Queue",
q:"An infinite set {1,2,3,…}. popSmallest removes and returns the smallest; addBack(x) restores x.",i:"pop, pop, addBack(1), pop, pop",o:"1, 2, 1, 3",
code:["class SmallestInfiniteSet:","    def __init__(self):","        self.cur, self.heap, self.inHeap = 1, [], set()","    def popSmallest(self):","        if self.heap:","            x = heappop(self.heap); self.inHeap.remove(x); return x","        self.cur += 1; return self.cur - 1","    def addBack(self, num):","        if num < self.cur and num not in self.inHeap:","            heappush(self.heap, num); self.inHeap.add(num)"],
time:"O(log n) per op",space:"O(n)",explain:"A counter for the untouched tail plus a heap of returned numbers.",best:"Counter for 1,2,3… plus a min-heap for numbers added back.",
notes:["Only numbers below cur can be added back.","The set avoids duplicate heap entries."],
frames:L=>[F("Init","cur = 1, heap empty.",[L("self.cur, self.heap")],[R("heap",[E])],{cur:1}),
F("popSmallest()","Heap empty → return cur (1), cur → 2.",[L("self.cur += 1")],[R("heap",[E])],{returned:1,cur:2}),
F("popSmallest()","Return 2, cur → 3.",[L("self.cur += 1")],[R("heap",[E])],{returned:2,cur:3}),
F("addBack(1)","1 < cur → push to heap.",[L("if num < self.cur"),L("heappush")],[R("heap",[1],[0])],{cur:3}),
F("popSmallest()","Heap not empty → pop 1.",[L("x = heappop")],[R("heap",[E])],{returned:1,cur:3}),
F("popSmallest()","Heap empty → return 3.",[L("self.cur += 1")],[R("heap",[E])],{returned:3,cur:4})]});
// 2542
prob({id:2542,slug:"maximum-subsequence-score",title:"Maximum Subsequence Score",level:"Medium",topic:"Heap / Priority Queue",
q:"Pick k indices; score = (sum of nums1) × (min of nums2). Maximise it.",i:"nums1=[1,3,3,2], nums2=[2,1,3,4], k=3",o:"12",
code:["class Solution:","    def maxScore(self, nums1, nums2, k):","        pairs = sorted(zip(nums2, nums1), reverse=True)","        heap, total, best = [], 0, 0","        for n2, n1 in pairs:","            heappush(heap, n1); total += n1","            if len(heap) > k: total -= heappop(heap)","            if len(heap) == k: best = max(best, total * n2)","        return best"],
time:"O(n log n)",space:"O(k)",explain:"Fix the min via sorted nums2; keep the k biggest nums1.",best:"Sort by nums2 desc so the current n2 is the min; a min-heap keeps the k largest nums1.",
notes:["Descending nums2 → current n2 is the minimum so far.","Drop the smallest nums1 to keep the sum large."],
frames:L=>{const pairs=[[4,2],[3,3],[2,1],[1,3]],h=[];let tot=0,best=0;const f=[F("Sort","Pairs (nums2, nums1) by nums2 descending.",[L("pairs = sorted")],[R("(n2, n1)",pairs.map(p=>"("+p+")"))])];
 pairs.forEach(([n2,n1],i)=>{h.push(n1);h.sort((a,b)=>a-b);tot+=n1;let d="Add "+n1+".";if(h.length>3){tot-=h.shift();d+=" Drop the smallest."}if(h.length===3){best=Math.max(best,tot*n2);d+=" Score "+tot+"×"+n2+" = "+tot*n2+"."}
 f.push(F("n2 = "+n2,d,[L("heappush"),L("if len(heap) > k"),L("if len(heap) == k")],[R("(n2, n1)",pairs.map(p=>"("+p+")"),[i]),R("heap",h,[0])],{total:tot,best}))});
 f.push(F("Return","Best score = 12.",[L("return best")],[R("answer",[best],[0])]));return f}});
// 2462
prob({id:2462,slug:"total-cost-to-hire-k-workers",title:"Total Cost to Hire K Workers",level:"Medium",topic:"Heap / Priority Queue",
q:"Hire k workers in rounds; each round take the cheapest among the first/last `candidates` (ties → lower index).",i:"[17,12,10,2,7,2,11,20,8], k=3, candidates=4",o:"11",
code:["class Solution:","    def totalCost(self, costs, k, candidates):","        n = len(costs)","        left = costs[:candidates]; heapify(left)","        right = costs[max(candidates, n - candidates):]; heapify(right)","        i, j, total = candidates, n - candidates - 1, 0","        for _ in range(k):","            if not right or (left and left[0] <= right[0]):","                total += heappop(left)","                if i <= j: heappush(left, costs[i]); i += 1","            else:","                total += heappop(right)","                if i <= j: heappush(right, costs[j]); j -= 1","        return total"],
time:"O((k+c) log c)",space:"O(c)",explain:"Two heaps hold the front and back candidate windows.",best:"Two min-heaps for the two ends; refill from the middle after each hire.",
notes:["Ties go to the left heap (lower index).","i and j are the next unseen middle workers."],
frames:L=>{let left=[2,10,12,17],right=[2,8,11,20],i=4,j=4,t=0;const f=[F("Two heaps","Left: first 4. Right: last 4.",[L("left = costs"),L("right = costs")],[R("left heap",left),R("right heap",right)])];
 let r1="Left 2 ≤ right 2 → hire left. Refill with costs[4] = 7.",r2="Right 2 < left 7 → hire right. Nothing left to refill.",r3="Left 7 < right 8 → hire left.";
 left=[7,10,12,17];t=2;f.push(F("Hire 1",r1,[L("left[0] <= right[0]"),L("total += heappop(left)")],[R("left heap",left,[0]),R("right heap",right)],{total:t}));
 right=[8,11,20];t=4;f.push(F("Hire 2",r2,[L("total += heappop(right)")],[R("left heap",left),R("right heap",right,[0])],{total:t}));
 left=[10,12,17];t=11;f.push(F("Hire 3",r3,[L("total += heappop(left)")],[R("left heap",left,[0]),R("right heap",right)],{total:t}));
 f.push(F("Return","2 + 2 + 7 = 11",[L("return total")],[R("answer",[11],[0])]));return f}});
// 374
prob({id:374,slug:"guess-number-higher-or-lower",title:"Guess Number Higher or Lower",level:"Easy",topic:"Binary Search",
q:"Guess the number I picked from 1…n using an API that says higher or lower.",i:"n = 10, picked = 6",o:"6",
code:["class Solution:","    def guessNumber(self, n):","        lo, hi = 1, n","        while lo <= hi:","            mid = (lo + hi) // 2","            g = guess(mid)","            if g == 0: return mid","            if g < 0: hi = mid - 1","            else: lo = mid + 1"],
time:"O(log n)",space:"O(1)",explain:"Halve the range each guess.",best:"Classic binary search on the range 1…n.",
notes:["guess(mid) = -1 means the pick is lower than mid.","Each call halves the search space."],
frames:L=>{const a=range(1,10),f=[F("Range","Pick is hidden in 1…10.",[L("lo, hi")],[R("n",a,[],M({lo:0,hi:9}))])];let lo=1,hi=10;
 while(lo<=hi){const mid=(lo+hi)>>1;const g=mid===6?0:(6<mid?-1:1);const d=g===0?"Correct!":g<0?"Too high → hi = "+(mid-1):"Too low → lo = "+(mid+1);
 f.push(F("mid = "+mid,d,[L("mid ="),L(g===0?"return mid":g<0?"if g < 0":"else: lo")],[R("n",a,[mid-1],M({lo:lo-1,hi:hi-1,mid:mid-1}))]));if(g===0)break;if(g<0)hi=mid-1;else lo=mid+1}return f}});
// 2300
prob({id:2300,slug:"successful-pairs-of-spells-and-potions",title:"Successful Pairs of Spells and Potions",level:"Medium",topic:"Binary Search",
q:"A pair succeeds if spell × potion ≥ success. For each spell, count the potions that work.",i:"spells=[5,1,3], potions=[1,2,3,4,5], success=7",o:"[4, 0, 3]",
code:["class Solution:","    def successfulPairs(self, spells, potions, success):","        potions.sort()","        m = len(potions)","        return [m - bisect_left(potions, (success + s - 1) // s) for s in spells]"],
time:"O((n+m) log m)",space:"O(1)",explain:"Binary search the smallest potion that is strong enough.",best:"Sort potions once; bisect for the cut-off per spell.",
notes:["need = ceil(success / s) via (success + s - 1) // s.","Everything right of the cut-off works."],
frames:L=>{const p=[1,2,3,4,5];return [F("Sorted potions","For each spell, find the first potion ≥ need.",[L("potions.sort()")],[R("potions",p)]),
F("Spell 5","need = ceil(7/5) = 2 → first at index 1 → 5 − 1 = 4 pairs.",[L("bisect_left")],[R("potions",p,[1,2,3,4],M({need:1}))],{count:4}),
F("Spell 1","need = 7 → none (index 5) → 0 pairs.",[L("bisect_left")],[R("potions",p,[],{})],{count:0}),
F("Spell 3","need = ceil(7/3) = 3 → index 2 → 5 − 2 = 3 pairs.",[L("bisect_left")],[R("potions",p,[2,3,4],M({need:2}))],{count:3}),
F("Return","[4, 0, 3]",[L("return")],[R("answer",[4,0,3],[0,1,2])])]}});
// 162
prob({id:162,slug:"find-peak-element",title:"Find Peak Element",level:"Medium",topic:"Binary Search",
q:"Find any index whose value is greater than both neighbours, in O(log n).",i:"[1, 2, 1, 3, 5, 6, 4]",o:"5 (index of 6)",
code:["class Solution:","    def findPeakElement(self, nums):","        lo, hi = 0, len(nums) - 1","        while lo < hi:","            mid = (lo + hi) // 2","            if nums[mid] < nums[mid + 1]: lo = mid + 1","            else: hi = mid","        return lo"],
time:"O(log n)",space:"O(1)",explain:"Follow the uphill side.",best:"Binary search on slope: a peak always exists in the rising direction.",
notes:["If nums[mid] < nums[mid+1], a peak lies to the right.","Otherwise mid itself or the left holds one."],
frames:L=>{const a=[1,2,1,3,5,6,4],f=[F("Range","Search whole array.",[L("lo, hi")],[R("nums",a,[],M({lo:0,hi:6}))])];let lo=0,hi=6;
 while(lo<hi){const mid=(lo+hi)>>1,up=a[mid]<a[mid+1];f.push(F("mid = "+mid,a[mid]+(up?" < ":" > ")+a[mid+1]+(up?" → rising, go right":" → falling, go left"),[L("mid ="),L(up?"lo = mid + 1":"else: hi")],[R("nums",a,[mid,mid+1],M({lo,hi,mid}))]));if(up)lo=mid+1;else hi=mid}
 f.push(F("Return","lo = hi = "+lo+" → value 6.",[L("return lo")],[R("nums",a,[lo],M({peak:lo}))]));return f}});
// 875
prob({id:875,slug:"koko-eating-bananas",title:"Koko Eating Bananas",level:"Medium",topic:"Binary Search",
q:"Smallest eating speed k (bananas/hour) to finish all piles within h hours.",i:"piles=[3,6,7,11], h=8",o:"4",
code:["class Solution:","    def minEatingSpeed(self, piles, h):","        lo, hi = 1, max(piles)","        while lo < hi:","            mid = (lo + hi) // 2","            if sum(ceil(p / mid) for p in piles) <= h: hi = mid","            else: lo = mid + 1","        return lo"],
time:"O(n log max)",space:"O(1)",explain:"Binary search the answer (speed), not an index.",best:"Binary search on the answer: faster speed always works if a slower one does.",
notes:["hours(k) = Σ ceil(pile / k).","Feasibility is monotonic in k."],
frames:L=>{const sp=range(1,11),piles=[3,6,7,11],f=[F("Speeds","Speed is between 1 and the biggest pile (11).",[L("lo, hi")],[R("speed",sp,[],M({lo:0,hi:10}))])];let lo=1,hi=11;
 while(lo<hi){const mid=(lo+hi)>>1,hrs=piles.reduce((s,p)=>s+Math.ceil(p/mid),0),ok=hrs<=8;f.push(F("mid = "+mid,"Hours needed = "+hrs+(ok?" ≤ 8 → try slower":" > 8 → need faster"),[L("mid ="),L(ok?"hi = mid":"else: lo")],[R("speed",sp,[mid-1],M({lo:lo-1,hi:hi-1,mid:mid-1}))],{hours:hrs}));if(ok)hi=mid;else lo=mid+1}
 f.push(F("Return","Minimum speed = "+lo+".",[L("return lo")],[R("speed",sp,[lo-1],M({answer:lo-1}))]));return f}});
// 17
prob({id:17,slug:"letter-combinations-of-a-phone-number",title:"Letter Combinations of a Phone Number",level:"Medium",topic:"Backtracking",
q:"Return all letter combinations a digit string could represent on a phone keypad.",i:'"23"',o:'["ad","ae","af","bd","be","bf","cd","ce","cf"]',
code:["class Solution:","    def letterCombinations(self, digits):","        if not digits: return []","        phone = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}","        res = []","        def go(i, path):","            if i == len(digits): res.append(path); return","            for ch in phone[digits[i]]: go(i + 1, path + ch)","        go(0, '')","        return res"],
time:"O(4ⁿ·n)",space:"O(n)",explain:"Choose one letter per digit, recurse, backtrack.",best:"Backtracking: one level per digit, one branch per letter.",
notes:["Depth = number of digits.","path + ch builds a new string, so no explicit undo is needed."],
frames:L=>{const res=[],f=[F("Digits","2 → abc, 3 → def",[L("phone = ")],[R("digits","23"),R("res",[E])])];
 for(const a of "abc"){const add=[...'def'].map(b=>a+b);res.push(...add);f.push(F("Pick '"+a+"'","Then try d, e, f for digit 3.",[L("for ch in phone"),L("res.append")],[R("path",[a],[0]),R("res",res,range(res.length-3,res.length-1))]))}
 f.push(F("Return","9 combinations.",[L("return res")],[R("res",res,range(0,8))]));return f}});
// 216
prob({id:216,slug:"combination-sum-iii",title:"Combination Sum III",level:"Medium",topic:"Backtracking",
q:"Choose k distinct numbers from 1–9 that add up to n.",i:"k = 3, n = 7",o:"[[1,2,4]]",
code:["class Solution:","    def combinationSum3(self, k, n):","        res = []","        def go(start, path, total):","            if len(path) == k:","                if total == n: res.append(path[:])","                return","            for x in range(start, 10):","                if total + x > n: break","                path.append(x); go(x + 1, path, total + x); path.pop()","        go(1, [], 0)","        return res"],
time:"O(C(9,k)·k)",space:"O(k)",explain:"Increasing choices avoid duplicates; prune when over n.",best:"Backtrack in increasing order and prune once the sum exceeds n.",
notes:["start = x + 1 keeps numbers distinct and ordered.","break (not continue) because later x are larger."],
frames:L=>[F("Start","Choose 3 numbers from 1–9 summing to 7.",[L("go(1,")],[R("path",[E])]),
F("Build [1, 2]","Sum 3; need one more.",[L("path.append")],[R("path",[1,2],[0,1])],{total:3}),
F("Try 3 → [1,2,3]","Size 3 but sum 6 ≠ 7 → reject.",[L("if len(path) == k"),L("if total == n")],[R("path",[1,2,3],[2])],{total:6}),
F("Try 4 → [1,2,4]","Sum 7 = n → save!",[L("res.append")],[R("path",[1,2,4],[0,1,2]),R("res",["[1,2,4]"],[0])],{total:7}),
F("Try 5","3 + 5 = 8 > 7 → prune (break).",[L("if total + x > n")],[R("path",[1,2,5],[2])],{total:8}),
F("Return","Every other start is too big.",[L("return res")],[R("res",["[1,2,4]"],[0])])]});
