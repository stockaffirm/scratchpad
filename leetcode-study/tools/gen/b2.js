const {prob,R,M,F,range}=require("./lib");
prob({id:2390,slug:"removing-stars",title:"Removing Stars From a String",level:"Medium",topic:"Stack",
q:"Each * deletes the closest non-star letter to its left. Return what remains.",i:'"leet**cod*e"',o:'"lecoe"',
code:["class Solution:","    def removeStars(self, s):","        stack = []","        for ch in s:","            if ch == '*': stack.pop()","            else: stack.append(ch)","        return ''.join(stack)"],
time:"O(n)",space:"O(n)",explain:"A star pops the latest letter.",best:"A stack: the closest letter to the left is always the top.",
notes:["'Closest to the left' = last in, first out.","Join the stack once at the end."],
frames:L=>{const s="leet**cod*e",st=[],f=[F("Input","Scan left to right with a stack.",[L("stack = []")],[R("s",s),R("stack",["·"])])];
 [...s].forEach((c,i)=>{if(c==="*"){const p=st.pop();f.push(F("'*'","Pop '"+p+"'.",[L("if ch ==")],[R("s",s,[i],M({i})),R("stack",st.length?st:["·"],[])]))}else{st.push(c);f.push(F("'"+c+"'","Push.",[L("stack.append")],[R("s",s,[i],M({i})),R("stack",st,[st.length-1])]))}});
 f.push(F("Return","Join the stack.",[L("return")],[R("answer","lecoe",range(0,4))]));return f}});
prob({id:735,slug:"asteroid-collision",title:"Asteroid Collision",level:"Medium",topic:"Stack",
q:"Positive moves right, negative moves left. Smaller explodes; equal both explode. Final state?",i:"[5, 10, -5, -12]",o:"[-12]",
code:["class Solution:","    def asteroidCollision(self, asteroids):","        stack = []","        for a in asteroids:","            while stack and a < 0 < stack[-1]:","                if stack[-1] < -a:","                    stack.pop(); continue","                if stack[-1] == -a: stack.pop()","                break","            else:","                stack.append(a)","        return stack"],
time:"O(n)",space:"O(n)",explain:"Only a left-mover meeting a right-mover on top collides.",best:"Stack of survivors; collisions only happen at the top.",
notes:["Collision needs stack top > 0 and new a < 0.","while/else: else runs only if no break."],
frames:L=>{const A=[5,10,-5,-12],st=[],f=[F("Input","5 and 10 go right; -5 and -12 go left.",[L("stack = []")],[R("asteroids",A),R("stack",["·"])])];
 A.forEach((a,i)=>{let d="";let alive=true;while(st.length&&a<0&&st[st.length-1]>0){const t=st[st.length-1];if(t<-a){st.pop();d+="Top "+t+" destroyed. ";continue}if(t===-a){st.pop();d+="Both explode. "}else d+="New "+a+" destroyed. ";alive=false;break}
 if(alive)st.push(a);f.push(F("a = "+a,(d||"No collision. ")+(alive?"Push "+a+".":""),[L("while stack and"),L("stack.append")],[R("asteroids",A,[i],M({a:i})),R("stack",st.length?st:["·"],alive?[st.length-1]:[])]))});
 f.push(F("Return","Survivors.",[L("return")],[R("answer",st,range(0,st.length-1))]));return f}});
prob({id:394,slug:"decode-string",title:"Decode String",level:"Medium",topic:"Stack",
q:"Expand k[encoded] patterns, possibly nested.",i:'"3[a2[c]]"',o:'"accaccacc"',
code:["class Solution:","    def decodeString(self, s):","        stack, cur, num = [], '', 0","        for ch in s:","            if ch.isdigit(): num = num * 10 + int(ch)","            elif ch == '[':","                stack.append((cur, num)); cur, num = '', 0","            elif ch == ']':","                prev, k = stack.pop(); cur = prev + cur * k","            else: cur += ch","        return cur"],
time:"O(output)",space:"O(depth)",explain:"Push state at '[', pop and repeat at ']'.",best:"Stack saves the outer string and repeat count at each '['.",
notes:["Multi-digit numbers: num = num*10 + digit.","Each ']' repeats cur and glues it to the saved prefix."],
frames:L=>{const s="3[a2[c]]",st=[];let cur="",num=0;const f=[F("Input","Nested repeat counts.",[L("stack, cur")],[R("s",s)])];
 [...s].forEach((c,i)=>{let ln,d;if(/\d/.test(c)){num=num*10+ +c;ln=L("isdigit");d="Digit: num = "+num}else if(c==="["){st.push([cur,num]);d="Save ('"+cur+"', "+num+") on stack";cur="";num=0;ln=L("append((cur")}else if(c==="]"){const[p,k]=st.pop();const nc=p+cur.repeat(k);d="Pop → '"+p+"' + '"+cur+"'×"+k+" = '"+nc+"'";cur=nc;ln=L("prev, k")}else{cur+=c;d="Letter: cur = '"+cur+"'";ln=L("else: cur")}
 f.push(F("'"+c+"'",d,[ln],[R("s",s,[i],M({i})),R("stack",st.length?st.map(x=>"("+x[0]+","+x[1]+")"):["·"]),R("cur",cur?cur:["·"])],{num}))});
 f.push(F("Return","Fully expanded.",[L("return cur")],[R("answer",cur,range(0,cur.length-1))]));return f}});
prob({id:933,slug:"number-of-recent-calls",title:"Number of Recent Calls",level:"Easy",topic:"Queue",
q:"ping(t) returns how many pings happened in the last 3000 ms (inclusive).",i:"ping(1), ping(100), ping(3001), ping(3002)",o:"1, 2, 3, 3",
code:["class RecentCounter:","    def __init__(self):","        self.q = deque()","    def ping(self, t):","        self.q.append(t)","        while self.q[0] < t - 3000:","            self.q.popleft()","        return len(self.q)"],
time:"O(1) amortised",space:"O(window)",explain:"Queue of timestamps; drop the expired ones from the front.",best:"A queue: times arrive in order, so old ones expire from the front.",
notes:["Each timestamp is pushed and popped once.","Window is [t-3000, t], inclusive."],
frames:L=>{const q=[],f=[F("Empty queue","Timestamps arrive in increasing order.",[L("self.q = deque")],[R("queue",["·"])])];
 for(const t of [1,100,3001,3002]){q.push(t);const gone=[];while(q[0]<t-3000)gone.push(q.shift());
 f.push(F("ping("+t+")",gone.length?"Window starts at "+(t-3000)+": drop "+gone.join(", ")+".":"Nothing expired (cut-off "+(t-3000)+").",[L("self.q.append"),L("while self.q[0]"),L("return len")],[R("queue",q,[q.length-1],{},"list")],{"cut-off":t-3000,count:q.length}))}return f}});
prob({id:649,slug:"dota2-senate",title:"Dota2 Senate",level:"Medium",topic:"Queue",
q:"Senators ban opponents in turn. Which party wins?",i:'"RDD"',o:'"Dire"',
code:["class Solution:","    def predictPartyVictory(self, senate):","        n = len(senate)","        R = deque(i for i, c in enumerate(senate) if c == 'R')","        D = deque(i for i, c in enumerate(senate) if c == 'D')","        while R and D:","            r, d = R.popleft(), D.popleft()","            if r < d: R.append(r + n)","            else: D.append(d + n)","        return 'Radiant' if R else 'Dire'"],
time:"O(n)",space:"O(n)",explain:"Earlier index acts first, bans the other, and re-queues for the next round.",best:"Two queues of indices; the smaller index wins and goes to the back (+n).",
notes:["Index + n = the senator's slot in the next round.","The queue that empties loses."],
frames:L=>[F("Queues","R at 0; D at 1 and 2.",[L("R = deque"),L("D = deque")],[R("R",[0],[],{},"list"),R("D",[1,2],[],{},"list")]),
F("Round: 0 vs 1","R(0) acts first → bans D(1). R re-queues as 3.",[L("r, d ="),L("if r < d")],[R("R",[3],[0],{},"list"),R("D",[2],[],{},"list")]),
F("Round: 3 vs 2","D(2) acts first → bans R(3). D re-queues as 5.",[L("r, d ="),L("else: D.append")],[R("R",["·"],[]),R("D",[5],[0],{},"list")]),
F("Return","R is empty → Dire wins.",[L("return")],[R("winner",["Dire"],[0])])]});
prob({id:2095,slug:"delete-the-middle-node",title:"Delete the Middle Node of a Linked List",level:"Medium",topic:"Linked List",
q:"Delete the middle node (index ⌊n/2⌋) and return the head.",i:"1 → 3 → 4 → 7 → 1 → 2 → 6",o:"1 → 3 → 4 → 1 → 2 → 6",
code:["class Solution:","    def deleteMiddle(self, head):","        if not head.next: return None","        slow, fast = head, head.next.next","        while fast and fast.next:","            slow, fast = slow.next, fast.next.next","        slow.next = slow.next.next","        return head"],
time:"O(n)",space:"O(1)",explain:"fast moves twice as fast; slow stops just before the middle.",best:"Slow/fast pointers; start fast one step ahead so slow lands before the middle.",
notes:["Starting fast at head.next.next makes slow the node before the middle.","Single-node list returns None."],
frames:L=>{const v=[1,3,4,7,1,2,6],n=7,f=[F("Start","slow at 0, fast at 2.",[L("slow, fast = head")],[R("list",v,[0,2],M({slow:0,fast:2}),"list")])];let s=0,fa=2;
 while(fa<n&&fa+1<n){s++;fa+=2;f.push(F("Advance","slow +1, fast +2.",[L("while fast and"),L("slow, fast = slow.next")],[R("list",v,[s],M({slow:s,fast:fa<n?fa:null}),"list")]))}
 f.push(F("Unlink","slow.next jumps over the middle (7).",[L("slow.next = slow.next.next")],[R("list",v,[2,3],M({slow:2,skip:3}),"list")]));
 f.push(F("Return","Middle removed.",[L("return head")],[R("list",[1,3,4,1,2,6],[],{},"list")]));return f}});
prob({id:328,slug:"odd-even-linked-list",title:"Odd Even Linked List",level:"Medium",topic:"Linked List",
q:"Group all odd-position nodes first, then even-position nodes, in place.",i:"1 → 2 → 3 → 4 → 5",o:"1 → 3 → 5 → 2 → 4",
code:["class Solution:","    def oddEvenList(self, head):","        if not head: return head","        odd, even, evenHead = head, head.next, head.next","        while even and even.next:","            odd.next = even.next; odd = odd.next","            even.next = odd.next; even = even.next","        odd.next = evenHead","        return head"],
time:"O(n)",space:"O(1)",explain:"Weave two chains, then attach even after odd.",best:"Build the odd and even chains in one pass, then join them.",
notes:["Keep evenHead to reattach later.","Loop while even and even.next exist."],
frames:L=>[F("Input","Positions 1,3,5 are odd; 2,4 are even.",[L("def odd")],[R("list",[1,2,3,4,5],[0,2,4],{},"list")]),
F("Two heads","odd = 1, even = evenHead = 2.",[L("odd, even, evenHead")],[R("odd chain",[1],[0],{},"list"),R("even chain",[2],[0],{},"list")]),
F("Round 1","odd takes 3; even takes 4.",[L("odd.next = even.next"),L("even.next = odd.next")],[R("odd chain",[1,3],[1],{},"list"),R("even chain",[2,4],[1],{},"list")]),
F("Round 2","odd takes 5; even.next is None → stop.",[L("odd.next = even.next"),L("while even")],[R("odd chain",[1,3,5],[2],{},"list"),R("even chain",[2,4],[],{},"list")]),
F("Connect","odd tail → evenHead.",[L("odd.next = evenHead")],[R("result",[1,3,5,2,4],[3,4],{},"list")])]});
prob({id:206,slug:"reverse-linked-list",title:"Reverse Linked List",level:"Easy",topic:"Linked List",
q:"Reverse a singly linked list.",i:"1 → 2 → 3 → 4 → 5",o:"5 → 4 → 3 → 2 → 1",
code:["class Solution:","    def reverseList(self, head):","        prev, curr = None, head","        while curr:","            nxt = curr.next","            curr.next = prev","            prev, curr = curr, nxt","        return prev"],
time:"O(n)",space:"O(1)",explain:"Flip one pointer per step.",best:"Iterative three-pointer reversal in O(1) space.",
notes:["Save next before overwriting curr.next.","prev is the new head at the end."],
frames:L=>{const f=[F("Start","prev = None, curr = 1.",[L("prev, curr")],[R("reversed",["·"]),R("remaining",[1,2,3,4,5],[0],{},"list")])];
 for(let k=1;k<=5;k++){f.push(F("Flip "+k,"Point node "+k+" back at the reversed part.",[L("nxt = "),L("curr.next = prev"),L("prev, curr =")],[R("reversed",range(1,k).reverse(),[0],{},"list"),R("remaining",k<5?range(k+1,5):["·"],k<5?[0]:[],{},"list")]))}
 f.push(F("Return","prev is the new head.",[L("return prev")],[R("result",[5,4,3,2,1],[0],{},"list")]));return f}});
prob({id:2130,slug:"maximum-twin-sum",title:"Maximum Twin Sum of a Linked List",level:"Medium",topic:"Linked List",
q:"Node i pairs with node n-1-i (its twin). Find the largest twin sum.",i:"4 → 2 → 2 → 3",o:"7",
code:["class Solution:","    def pairSum(self, head):","        vals = []","        while head:","            vals.append(head.val); head = head.next","        return max(vals[i] + vals[-1 - i] for i in range(len(vals) // 2))"],
time:"O(n)",space:"O(n)",explain:"Copy to an array, then pair ends inward.",best:"Array copy is simplest; reversing the second half gives O(1) space.",
notes:["Twins: index i and n-1-i.","Only n/2 pairs to check."],
frames:L=>[F("List","Walk the list and copy values.",[L("vals = []"),L("while head")],[R("vals",[4,2,2,3],[],{},"list")]),
F("Pair 0","4 + 3 = 7",[L("return max")],[R("vals",[4,2,2,3],[0,3],M({i:0,twin:3}))],{sum:7}),
F("Pair 1","2 + 2 = 4",[L("return max")],[R("vals",[4,2,2,3],[1,2],M({i:1,twin:2}))],{sum:4,max:7}),
F("Return","Largest twin sum = 7.",[L("return max")],[R("answer",[7],[0])])]});
