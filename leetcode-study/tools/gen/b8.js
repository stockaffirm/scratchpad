const {prob,R,M,F,range}=require("./lib");const E="·";
const LL=(n,c,h=[],m={})=>R(n,c.length?c:[E],h,m,c.length?"list":undefined);
prob({id:20,slug:"valid-parentheses",title:"Valid Parentheses",level:"Easy",topic:"Stack",
x:["Is every bracket ( [ { closed by the matching bracket in the right order?","'([{}])' is perfectly nested → True. '(]' would be False.","Push opening brackets. On a closing bracket, the top of the stack must be its partner."],
i:'"([{}])"',o:"True",
code:["class Solution:","    def isValid(self, s):","        stack, pairs = [], {')':'(', ']':'[', '}':'{'}","        for ch in s:","            if ch in pairs:","                if not stack or stack.pop() != pairs[ch]: return False","            else: stack.append(ch)","        return not stack"],
time:"O(n)",space:"O(n)",
frames:L=>{const s="([{}])",st=[],f=[F("Start","Empty stack.",[L("stack, pairs")],[R("s",s),R("stack",[E])])];
 [...s].forEach((c,i)=>{if("([{".includes(c)){st.push(c);f.push(F("'"+c+"'","Opening → push.",[L("else: stack.append")],[R("s",s,[i],M({i})),R("stack",st,[st.length-1])]))}else{const t=st.pop();f.push(F("'"+c+"'","Closing → pops '"+t+"' (its partner).",[L("if ch in pairs"),L("stack.pop()")],[R("s",s,[i],M({i})),R("stack",st.length?st:[E])]))}});
 f.push(F("Return","Stack empty → True.",[L("return not stack")],[R("answer",["True"],[0])]));return f}});
prob({id:71,slug:"simplify-path",title:"Simplify Path",level:"Medium",topic:"Stack",
x:["Clean up a Unix file path: '.' means stay, '..' means go up one folder, and extra slashes collapse.","'/a/./b/../../c/' → into a, stay, into b, up, up, into c → '/c'.","Split on '/'; push folder names, pop on '..'."],
i:'"/a/./b/../../c/"',o:'"/c"',
code:["class Solution:","    def simplifyPath(self, path):","        stack = []","        for part in path.split('/'):","            if part == '..':","                if stack: stack.pop()","            elif part and part != '.': stack.append(part)","        return '/' + '/'.join(stack)"],
time:"O(n)",space:"O(n)",
frames:L=>{const parts=["a",".","b","..","..","c"],st=[],f=[F("Split","Pieces between slashes.",[L("path.split")],[R("parts",parts)])];
 parts.forEach((p,i)=>{let d;if(p===".."){st.pop();d="'..' → go up (pop).";f.push(F(p,d,[L("if part == '..'")],[R("parts",parts,[i]),R("stack",st.length?st:[E])]))}else if(p!=="."){st.push(p);f.push(F(p,"Folder → push.",[L("elif part and")],[R("parts",parts,[i]),R("stack",st,[st.length-1])]))}else f.push(F(p,"'.' → stay, ignore.",[L("elif part and")],[R("parts",parts,[i]),R("stack",st.length?st:[E])]))});
 f.push(F("Return","'/' + 'c'",[L("return '/'")],[R("answer","/c")]));return f}});
prob({id:155,slug:"min-stack",title:"Min Stack",level:"Medium",topic:"Stack",
x:["A stack that can also tell you its smallest value instantly.","push -2, 0, -3 → getMin = -3 → pop → top = 0 → getMin = -2.","Store each value together with the minimum so far."],
i:"push(-2), push(0), push(-3), getMin(), pop(), top(), getMin()",o:"-3, 0, -2",
code:["class MinStack:","    def __init__(self): self.st = []","    def push(self, v):","        self.st.append((v, min(v, self.st[-1][1] if self.st else v)))","    def pop(self): self.st.pop()","    def top(self): return self.st[-1][0]","    def getMin(self): return self.st[-1][1]"],
time:"O(1) per op",space:"O(n)",
frames:L=>[F("push(-2)","min so far = -2.",[L("self.st.append")],[R("(value, min)",["(-2,-2)"],[0])]),F("push(0)","min stays -2.",[L("self.st.append")],[R("(value, min)",["(-2,-2)","(0,-2)"],[1])]),F("push(-3)","new min = -3.",[L("self.st.append")],[R("(value, min)",["(-2,-2)","(0,-2)","(-3,-3)"],[2])]),F("getMin()","Top's min → -3.",[L("def getMin")],[R("(value, min)",["(-2,-2)","(0,-2)","(-3,-3)"],[2])],{returns:-3}),F("pop()","Remove (-3,-3).",[L("self.st.pop()")],[R("(value, min)",["(-2,-2)","(0,-2)"],[1])]),F("top() / getMin()","top = 0, min = -2.",[L("def top")],[R("(value, min)",["(-2,-2)","(0,-2)"],[1])],{top:0,min:-2})]});
prob({id:150,slug:"evaluate-reverse-polish-notation",title:"Evaluate Reverse Polish Notation",level:"Medium",topic:"Stack",
x:["Evaluate math written with the operator AFTER its numbers (e.g. 2 1 + means 2 + 1).","['2','1','+','3','*'] = (2 + 1) × 3 = 9.","Push numbers; at an operator pop two, compute, push the result."],
i:'["2","1","+","3","*"]',o:"9",
code:["class Solution:","    def evalRPN(self, tokens):","        st = []","        for t in tokens:","            if t in '+-*/':","                b, a = st.pop(), st.pop()","                st.append(int(a + b if t == '+' else a - b if t == '-' else a * b if t == '*' else a / b))","            else: st.append(int(t))","        return st[0]"],
time:"O(n)",space:"O(n)",
frames:L=>{const tk=["2","1","+","3","*"],st=[],f=[F("Start","Empty stack.",[L("st = []")],[R("tokens",tk),R("stack",[E])])];
 tk.forEach((t,i)=>{if("+-*/".includes(t)){const b=st.pop(),a=st.pop(),r=t==="+"?a+b:t==="*"?a*b:t==="-"?a-b:Math.trunc(a/b);st.push(r);f.push(F("'"+t+"'",a+" "+t+" "+b+" = "+r,[L("b, a = st.pop()"),L("st.append(int(a")],[R("tokens",tk,[i]),R("stack",st,[st.length-1])]))}else{st.push(+t);f.push(F("'"+t+"'","Number → push.",[L("else: st.append")],[R("tokens",tk,[i]),R("stack",st,[st.length-1])]))}});
 f.push(F("Return","9",[L("return st[0]")],[R("answer",[9],[0])]));return f}});
prob({id:224,slug:"basic-calculator",title:"Basic Calculator",level:"Hard",topic:"Stack",
x:["Evaluate an expression with + , − and parentheses (no multiplication).","'1-(2+3)+4' = 1 − 5 + 4 = 0.","Track a running total and the current sign. On '(' save them on a stack; on ')' combine."],
i:'"1-(2+3)+4"',o:"0",
code:["class Solution:","    def calculate(self, s):","        total, num, sign, stack = 0, 0, 1, []","        for ch in s:","            if ch.isdigit(): num = num * 10 + int(ch)","            elif ch in '+-':","                total += sign * num; num = 0; sign = 1 if ch == '+' else -1","            elif ch == '(':","                stack.append((total, sign)); total, sign = 0, 1","            elif ch == ')':","                total += sign * num; num = 0","                prev, psign = stack.pop(); total = prev + psign * total","        return total + sign * num"],
time:"O(n)",space:"O(n)",
frames:L=>{const s="1-(2+3)+4";let total=0,num=0,sign=1;const st=[],f=[F("Start","total 0, sign +.",[L("total, num, sign")],[R("s",s)])];
 [...s].forEach((c,i)=>{let ln,d;if(/\d/.test(c)){num=num*10+ +c;ln=L("isdigit");d="Digit → num = "+num}else if(c==="+"||c==="-"){total+=sign*num;num=0;sign=c==="+"?1:-1;ln=L("total += sign * num; num = 0; sign");d="Add pending number; sign is now "+(sign>0?"+":"−")}else if(c==="("){st.push([total,sign]);d="Save (total "+total+", sign "+(sign>0?"+":"−")+"), start fresh";total=0;sign=1;ln=L("stack.append")}else{total+=sign*num;num=0;const[p,ps]=st.pop();total=p+ps*total;d="Close group: total = "+total;ln=L("prev, psign")}
 f.push(F("'"+c+"'",d,[ln],[R("s",s,[i],M({i})),R("stack",st.length?st.map(x=>"("+x+")"):[E])],{total,sign:sign>0?"+":"−"}))});
 total+=sign*num;f.push(F("Return","Add the last number → "+total+".",[L("return total + sign")],[R("answer",[total],[0])]));return f}});
prob({id:141,slug:"linked-list-cycle",title:"Linked List Cycle",level:"Easy",topic:"Linked List",
x:["Does the linked list loop back on itself?","3→2→0→−4 and −4 points back to 2 → there's a cycle → True.","A slow pointer (1 step) and a fast pointer (2 steps): in a loop, fast catches slow."],
i:"3 → 2 → 0 → -4 → (back to 2)",o:"True",
code:["class Solution:","    def hasCycle(self, head):","        slow = fast = head","        while fast and fast.next:","            slow, fast = slow.next, fast.next.next","            if slow is fast: return True","        return False"],
time:"O(n)",space:"O(1)",
frames:L=>{const v=[3,2,0,-4];return [F("Start","Both at the head. The tail points back to node 2.",[L("slow = fast = head")],[LL("list",v,[0],M({"slow,fast":0}))]),F("Step 1","slow → 2, fast → 0.",[L("slow, fast = slow.next")],[LL("list",v,[1,2],M({slow:1,fast:2}))]),F("Step 2","slow → 0, fast → 2 (it looped around).",[L("slow, fast = slow.next")],[LL("list",v,[2,1],M({slow:2,fast:1}))]),F("Step 3","Both land on -4 → same node!",[L("if slow is fast")],[LL("list",v,[3],M({"slow,fast":3}))]),F("Return","Cycle found → True.",[L("return True")],[R("answer",["True"],[0])])]}});
prob({id:2,slug:"add-two-numbers",title:"Add Two Numbers",level:"Medium",topic:"Linked List",
x:["Two numbers are stored as linked lists with digits in REVERSE order. Add them and return the sum the same way.","2→4→3 is 342, 5→6→4 is 465 → 807, stored as 7→0→8.","Add digit by digit like on paper, carrying over."],
i:"2→4→3 and 5→6→4",o:"7 → 0 → 8",
code:["class Solution:","    def addTwoNumbers(self, l1, l2):","        dummy = cur = ListNode(); carry = 0","        while l1 or l2 or carry:","            s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry","            carry, d = divmod(s, 10)","            cur.next = ListNode(d); cur = cur.next","            l1 = l1.next if l1 else None; l2 = l2.next if l2 else None","        return dummy.next"],
time:"O(max(n,m))",space:"O(max(n,m))",
frames:L=>[F("Start","Add digits from the least significant end.",[L("dummy = cur")],[LL("l1",[2,4,3]),LL("l2",[5,6,4]),LL("sum",[])]),F("2 + 5","= 7, carry 0.",[L("s ="),L("carry, d")],[LL("l1",[2,4,3],[0]),LL("l2",[5,6,4],[0]),LL("sum",[7],[0])],{carry:0}),F("4 + 6","= 10 → digit 0, carry 1.",[L("carry, d")],[LL("l1",[2,4,3],[1]),LL("l2",[5,6,4],[1]),LL("sum",[7,0],[1])],{carry:1}),F("3 + 4 + carry 1","= 8, carry 0.",[L("carry, d")],[LL("l1",[2,4,3],[2]),LL("l2",[5,6,4],[2]),LL("sum",[7,0,8],[2])],{carry:0}),F("Return","7 → 0 → 8 (807).",[L("return dummy.next")],[LL("answer",[7,0,8],[0,1,2])])]});
prob({id:21,slug:"merge-two-sorted-lists",title:"Merge Two Sorted Lists",level:"Easy",topic:"Linked List",
x:["Join two sorted linked lists into one sorted list.","[1,2,4] and [1,3,4] → 1,1,2,3,4,4.","Keep a tail pointer; repeatedly attach the smaller front node."],
i:"1→2→4 and 1→3→4",o:"1→1→2→3→4→4",
code:["class Solution:","    def mergeTwoLists(self, a, b):","        dummy = cur = ListNode()","        while a and b:","            if a.val <= b.val: cur.next, a = a, a.next","            else: cur.next, b = b, b.next","            cur = cur.next","        cur.next = a or b","        return dummy.next"],
time:"O(n+m)",space:"O(1)",
frames:L=>{const A=[1,2,4],B=[1,3,4];let i=0,j=0;const res=[],f=[F("Start","Compare the two fronts.",[L("dummy = cur")],[LL("a",A,[0]),LL("b",B,[0]),LL("merged",[])])];
 while(i<3&&j<3){if(A[i]<=B[j]){res.push(A[i]);f.push(F("Take "+A[i]+" from a","a front "+A[i]+" ≤ b front "+B[j],[L("if a.val <= b.val")],[LL("a",A,[i]),LL("b",B,[j]),LL("merged",res,[res.length-1])]));i++}else{res.push(B[j]);f.push(F("Take "+B[j]+" from b","b front "+B[j]+" < a front "+A[i],[L("else: cur.next, b")],[LL("a",A,[i]),LL("b",B,[j]),LL("merged",res,[res.length-1])]));j++}}
 const rest=A.slice(i).concat(B.slice(j));res.push(...rest);f.push(F("Attach the rest","One list is empty; append the other.",[L("cur.next = a or b")],[LL("merged",res,range(res.length-rest.length,res.length-1))]));return f}});
prob({id:138,slug:"copy-list-with-random-pointer",title:"Copy List with Random Pointer",level:"Medium",topic:"Linked List",
x:["Each node has a 'next' and also a 'random' pointer to any node. Make a deep copy of the whole list.","List A→B→C with random pointers; the copy has new nodes A'→B'→C' with matching pointers.","Pass 1: make a copy of every node and remember old→new in a dictionary. Pass 2: wire up next and random using the dictionary."],
i:"A→B→C, A.random=C, B.random=A, C.random=B",o:"A'→B'→C' with the same random links",
code:["class Solution:","    def copyRandomList(self, head):","        clone = {None: None}","        cur = head","        while cur:","            clone[cur] = Node(cur.val); cur = cur.next","        cur = head","        while cur:","            clone[cur].next = clone[cur.next]; clone[cur].random = clone[cur.random]; cur = cur.next","        return clone[head]"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Original","Three nodes with next and random pointers.",[L("def copyRandomList")],[LL("next",["A","B","C"],[]),R("random",["A→C","B→A","C→B"])]),F("Pass 1","Create A', B', C' with no pointers yet; remember A→A', B→B', C→C'.",[L("clone[cur] = Node")],[R("clone map",["A→A'","B→B'","C→C'"],[0,1,2])]),F("Pass 2","Set A'.next = clone[A.next] = B' and A'.random = clone[A.random] = C'. Same for the others.",[L("clone[cur].next")],[LL("copy next",["A'","B'","C'"],[0,1,2]),R("copy random",["A'→C'","B'→A'","C'→B'"],[0,1,2])]),F("Return","clone[head] = A'.",[L("return clone[head]")],[LL("copy",["A'","B'","C'"],[0])])]});
prob({id:92,slug:"reverse-linked-list-ii",title:"Reverse Linked List II",level:"Medium",topic:"Linked List",
x:["Reverse only the part of the list from position left to position right.","1→2→3→4→5, left 2, right 4 → 1→4→3→2→5.","Walk to the node before the section, then repeatedly move the next node to the front of the section."],
i:"1→2→3→4→5, left=2, right=4",o:"1→4→3→2→5",
code:["class Solution:","    def reverseBetween(self, head, left, right):","        dummy = ListNode(0, head); prev = dummy","        for _ in range(left - 1): prev = prev.next","        cur = prev.next","        for _ in range(right - left):","            nxt = cur.next","            cur.next = nxt.next","            nxt.next = prev.next","            prev.next = nxt","        return dummy.next"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","prev stops at 1 (just before position 2); cur = 2.",[L("for _ in range(left - 1)"),L("cur = prev.next")],[LL("list",[1,2,3,4,5],[1,2,3],M({prev:0,cur:1}))]),F("Move 3 to the front","nxt = 3 jumps in front of 2 (right after prev).",[L("nxt = cur.next"),L("prev.next = nxt")],[LL("list",[1,3,2,4,5],[1])]),F("Move 4 to the front","nxt = 4 jumps in front of 3.",[L("prev.next = nxt")],[LL("list",[1,4,3,2,5],[1])]),F("Return","Section 2–4 is reversed.",[L("return dummy.next")],[LL("list",[1,4,3,2,5],[1,2,3])])]});
prob({id:25,slug:"reverse-nodes-in-k-group",title:"Reverse Nodes in k-Group",level:"Hard",topic:"Linked List",
x:["Reverse the list k nodes at a time. If fewer than k nodes remain, leave them as they are.","1→2→3→4→5 with k = 2 → 2→1→4→3→5.","Check that k nodes exist, reverse them, and attach to the result of handling the rest."],
i:"1→2→3→4→5, k=2",o:"2→1→4→3→5",
code:["class Solution:","    def reverseKGroup(self, head, k):","        node, cnt = head, 0","        while node and cnt < k: node = node.next; cnt += 1","        if cnt < k: return head","        prev, cur = self.reverseKGroup(node, k), head","        for _ in range(k):","            cur.next, prev, cur = prev, cur, cur.next","        return prev"],
time:"O(n)",space:"O(n/k)",
frames:L=>[F("Start","k = 2: groups (1,2), (3,4), and a leftover 5.",[L("def reverseKGroup")],[LL("list",[1,2,3,4,5],[0,1])]),F("Group 1","Reverse 1,2 → 2,1; it attaches to whatever the rest becomes.",[L("for _ in range(k)")],[LL("list",[2,1,3,4,5],[0,1])]),F("Group 2","Reverse 3,4 → 4,3.",[L("for _ in range(k)")],[LL("list",[2,1,4,3,5],[2,3])]),F("Leftover","Only one node (5) left, fewer than k → unchanged.",[L("if cnt < k: return head")],[LL("list",[2,1,4,3,5],[4])])]});
prob({id:19,slug:"remove-nth-node-from-end-of-list",title:"Remove Nth Node From End of List",level:"Medium",topic:"Linked List",
x:["Remove the nth node counting from the END of the list.","1→2→3→4→5, n = 2 → remove 4 → 1→2→3→5.","Send fast n steps ahead, then move both until fast is at the end; slow is just before the target."],
i:"1→2→3→4→5, n=2",o:"1→2→3→5",
code:["class Solution:","    def removeNthFromEnd(self, head, n):","        dummy = ListNode(0, head); fast = slow = dummy","        for _ in range(n): fast = fast.next","        while fast.next:","            fast, slow = fast.next, slow.next","        slow.next = slow.next.next","        return dummy.next"],
time:"O(n)",space:"O(1)",
frames:L=>{const v=["D",1,2,3,4,5];return [F("Start","dummy node D in front; both pointers at D.",[L("dummy =")],[LL("list",v,[0],M({"slow,fast":0}))]),F("Gap of n","fast moves 2 steps ahead.",[L("for _ in range(n)")],[LL("list",v,[0,2],M({slow:0,fast:2}))]),F("Move together","Both advance until fast reaches the last node.",[L("while fast.next")],[LL("list",v,[3,5],M({slow:3,fast:5}))]),F("Unlink","slow.next (4) is skipped.",[L("slow.next = slow.next.next")],[LL("list",[1,2,3,5],[2])])]}});
prob({id:82,slug:"remove-duplicates-from-sorted-list-ii",title:"Remove Duplicates from Sorted List II",level:"Medium",topic:"Linked List",
x:["In a sorted linked list, delete EVERY value that appears more than once (not just the extras).","1→2→3→3→4→4→5 → 1→2→5.","Use a previous pointer; when a value repeats, skip all its copies."],
i:"1→2→3→3→4→4→5",o:"1→2→5",
code:["class Solution:","    def deleteDuplicates(self, head):","        dummy = prev = ListNode(0, head); cur = head","        while cur:","            if cur.next and cur.val == cur.next.val:","                while cur.next and cur.val == cur.next.val: cur = cur.next","                prev.next = cur.next","            else: prev = prev.next","            cur = cur.next","        return dummy.next"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","Walk with prev and cur.",[L("dummy = prev")],[LL("list",[1,2,3,3,4,4,5],[0],M({cur:0}))]),F("1, 2","Unique → prev moves along.",[L("else: prev = prev.next")],[LL("list",[1,2,3,3,4,4,5],[1])]),F("3, 3","Repeat → skip both 3s.",[L("while cur.next and"),L("prev.next = cur.next")],[LL("list",[1,2,3,3,4,4,5],[2,3])]),F("4, 4","Repeat → skip both 4s.",[L("prev.next = cur.next")],[LL("list",[1,2,4,4,5],[2,3])]),F("Return","5 is unique and stays.",[L("return dummy.next")],[LL("list",[1,2,5],[0,1,2])])]});
prob({id:61,slug:"rotate-list",title:"Rotate List",level:"Medium",topic:"Linked List",
x:["Rotate the linked list to the right by k places.","1→2→3→4→5, k = 2 → 4→5→1→2→3.","Make the list a circle, then cut it at the right spot."],
i:"1→2→3→4→5, k=2",o:"4→5→1→2→3",
code:["class Solution:","    def rotateRight(self, head, k):","        if not head or not head.next: return head","        n, tail = 1, head","        while tail.next: tail = tail.next; n += 1","        k %= n","        if k == 0: return head","        new_tail = head","        for _ in range(n - k - 1): new_tail = new_tail.next","        new_head = new_tail.next","        new_tail.next = None; tail.next = head","        return new_head"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Length","n = 5, k = 2 → new head is node 4 (n − k = 3 nodes stay in front).",[L("while tail.next"),L("k %= n")],[LL("list",[1,2,3,4,5],[4],M({tail:4}))]),F("Close the loop","tail.next = head.",[L("tail.next = head")],[LL("list",[1,2,3,4,5,"↺"],[4])]),F("Cut","Break after node 3; node 4 becomes the head.",[L("new_tail.next = None")],[LL("list",[4,5,1,2,3],[0])])]});
prob({id:86,slug:"partition-list",title:"Partition List",level:"Medium",topic:"Linked List",
x:["Rearrange so all nodes smaller than x come first, then the rest, keeping the original order within each group.","1→4→3→2→5→2 with x = 3 → 1→2→2→4→3→5.","Build two lists (small and big) and join them."],
i:"1→4→3→2→5→2, x=3",o:"1→2→2→4→3→5",
code:["class Solution:","    def partition(self, head, x):","        small = s = ListNode(); big = b = ListNode()","        while head:","            if head.val < x: s.next = head; s = s.next","            else: b.next = head; b = b.next","            head = head.next","        b.next = None; s.next = big.next","        return small.next"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[1,4,3,2,5,2],sm=[],bg=[],f=[F("Start","x = 3.",[L("small = s")],[LL("list",a)])];
 a.forEach((v,i)=>{if(v<3){sm.push(v);f.push(F(v+" < 3","→ small list.",[L("if head.val < x")],[LL("small",sm,[sm.length-1]),LL("big",bg)]))}else{bg.push(v);f.push(F(v+" ≥ 3","→ big list.",[L("else: b.next")],[LL("small",sm),LL("big",bg,[bg.length-1])]))}});
 f.push(F("Join","small followed by big.",[L("s.next = big.next")],[LL("result",sm.concat(bg),[0])]));return f}});
prob({id:146,slug:"lru-cache",title:"LRU Cache",level:"Medium",topic:"Linked List",
x:["A cache with a fixed size. When it's full, throw out the Least Recently Used item. get and put both count as 'using' an item.","Capacity 2: put 1, put 2, get 1, put 3 → evicts 2 (1 was used more recently), get 2 → −1.","Keep items in usage order (an ordered dictionary): move to the end when used, drop from the front when full."],
i:"cap=2: put(1), put(2), get(1), put(3), get(2)",o:"1, -1",
code:["class LRUCache:","    def __init__(self, capacity):","        self.cap, self.d = capacity, OrderedDict()","    def get(self, k):","        if k not in self.d: return -1","        self.d.move_to_end(k); return self.d[k]","    def put(self, k, v):","        self.d[k] = v; self.d.move_to_end(k)","        if len(self.d) > self.cap: self.d.popitem(last=False)"],
time:"O(1) per op",space:"O(capacity)",
frames:L=>[F("put(1), put(2)","Oldest on the left, newest on the right.",[L("self.d[k] = v")],[R("order (old → new)",["1","2"],[1])]),F("get(1)","1 is used → moves to the newest end.",[L("self.d.move_to_end(k); return")],[R("order (old → new)",["2","1"],[1])],{returns:1}),F("put(3)","Cache would hold 3 items → evict the oldest (2).",[L("self.d.popitem")],[R("order (old → new)",["1","3"],[1])]),F("get(2)","2 was evicted → −1.",[L("if k not in self.d")],[R("order (old → new)",["1","3"],[])],{returns:-1})]});
