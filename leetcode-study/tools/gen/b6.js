const {prob,R,M,F,range}=require("./lib");const E="·";
const T="Array / String";
prob({id:88,slug:"merge-sorted-array",title:"Merge Sorted Array",level:"Easy",topic:T,
x:["Two sorted lists. nums1 has empty room (zeros) at the end. Merge nums2 into nums1 so the whole thing is sorted, in place.","nums1 = [1,2,3,_,_,_], nums2 = [2,5,6] → [1,2,2,3,5,6].","Fill nums1 from the BACK with the biggest remaining value, so nothing gets overwritten."],
i:"nums1=[1,2,3,0,0,0], m=3, nums2=[2,5,6], n=3",o:"[1,2,2,3,5,6]",
code:["class Solution:","    def merge(self, nums1, m, nums2, n):","        i, j, k = m - 1, n - 1, m + n - 1","        while j >= 0:","            if i >= 0 and nums1[i] > nums2[j]:","                nums1[k] = nums1[i]; i -= 1","            else:","                nums1[k] = nums2[j]; j -= 1","            k -= 1"],
time:"O(m+n)",space:"O(1)",
frames:L=>{const a=[1,2,3,0,0,0],b=[2,5,6];let i=2,j=2,k=5;const f=[F("Start","Pointers at the ends: i (nums1 data), j (nums2), k (write spot).",[L("i, j, k")],[R("nums1",a,[],M({i,k})),R("nums2",b,[],M({j}))])];
 while(j>=0){let d;if(i>=0&&a[i]>b[j]){a[k]=a[i];d=a[i]+" (nums1) is bigger → write it at k.";i--}else{a[k]=b[j];d=b[j]+" (nums2) wins → write it at k.";j--}k--;f.push(F("Place at "+(k+1),d,[L("if i >= 0"),L("nums1[k] = nums1[i]")],[R("nums1",a,[k+1],M({i:i>=0?i:null,k:k>=0?k:null})),R("nums2",b,j>=0?[j]:[],M({j:j>=0?j:null}))]))}
 f.push(F("Done","nums2 is used up; nums1 is sorted.",[L("while j >= 0")],[R("nums1",a,range(0,5))]));return f}});
prob({id:27,slug:"remove-element",title:"Remove Element",level:"Easy",topic:T,
x:["Remove every copy of val from the list in place. Return how many items remain; the first k slots must hold them.","[3,2,2,3], val = 3 → keep 2,2 → return 2.","One pointer reads, another writes: copy only the values you want to keep."],
i:"nums=[3,2,2,3], val=3",o:"2 → [2,2,_,_]",
code:["class Solution:","    def removeElement(self, nums, val):","        k = 0","        for x in nums:","            if x != val:","                nums[k] = x; k += 1","        return k"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[3,2,2,3],src=a.slice(),f=[F("Start","k = next free slot.",[L("k = 0")],[R("nums",a,[],M({k:0}))])];let k=0;
 src.forEach((x,i)=>{if(x!==3){a[k]=x;k++;f.push(F("x = "+x,"Not 3 → keep it at k.",[L("if x != val"),L("nums[k] = x")],[R("nums",a,[k-1],M({k,read:i}))]))}else f.push(F("x = "+x,"Equals val → skip.",[L("if x != val")],[R("nums",a,[],M({k,read:i}))]))});
 f.push(F("Return","First k = "+k+" slots hold the answer.",[L("return k")],[R("nums",a.slice(0,k),range(0,k-1))]));return f}});
prob({id:26,slug:"remove-duplicates-from-sorted-array",title:"Remove Duplicates from Sorted Array",level:"Easy",topic:T,
x:["The list is sorted. Remove duplicates in place so each value appears once. Return how many unique values.","[0,0,1,1,1,2,2,3,3,4] → 0,1,2,3,4 → return 5.","Compare each number with the last kept one; keep it only if different."],
i:"[0,0,1,1,1,2,2,3,3,4]",o:"5 → [0,1,2,3,4,…]",
code:["class Solution:","    def removeDuplicates(self, nums):","        k = 1","        for i in range(1, len(nums)):","            if nums[i] != nums[k - 1]:","                nums[k] = nums[i]; k += 1","        return k"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[0,0,1,1,1,2,2,3,3,4],f=[F("Start","The first number is always kept; k = 1.",[L("k = 1")],[R("nums",a,[0],M({k:1}))])];let k=1;
 for(let i=1;i<a.length;i++){if(a[i]!==a[k-1]){a[k]=a[i];k++;f.push(F("i = "+i,a[i]+" is new → write at slot "+(k-1)+".",[L("if nums[i] !="),L("nums[k] = nums[i]")],[R("nums",a,[k-1,i],M({k,i}))]))}else f.push(F("i = "+i,a[i]+" is a repeat → skip.",[L("if nums[i] !=")],[R("nums",a,[i],M({k,i}))]))}
 f.push(F("Return","5 unique values.",[L("return k")],[R("nums",a.slice(0,k),range(0,k-1))]));return f}});
prob({id:80,slug:"remove-duplicates-from-sorted-array-ii",title:"Remove Duplicates from Sorted Array II",level:"Medium",topic:T,
x:["Sorted list. Allow each value at most TWICE. Remove extras in place and return the new length.","[1,1,1,2,2,3] → 1,1,2,2,3 → length 5.","Keep x if fewer than 2 are kept, or if it differs from the value two places back."],
i:"[1,1,1,2,2,3]",o:"5 → [1,1,2,2,3,_]",
code:["class Solution:","    def removeDuplicates(self, nums):","        k = 0","        for x in nums:","            if k < 2 or x != nums[k - 2]:","                nums[k] = x; k += 1","        return k"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[1,1,1,2,2,3],src=a.slice(),f=[F("Start","Allow up to two copies.",[L("k = 0")],[R("nums",a,[],M({k:0}))])];let k=0;
 src.forEach((x,i)=>{if(k<2||x!==a[k-2]){a[k]=x;k++;f.push(F("x = "+x,"Fewer than 2 copies kept → keep.",[L("if k < 2"),L("nums[k] = x")],[R("nums",a,[k-1],M({k,read:i}))]))}else f.push(F("x = "+x,"Already two copies of "+x+" → skip.",[L("if k < 2")],[R("nums",a,[],M({k,read:i}))]))});
 f.push(F("Return","Length "+k+".",[L("return k")],[R("nums",a.slice(0,k),range(0,k-1))]));return f}});
prob({id:169,slug:"majority-element",title:"Majority Element",level:"Easy",topic:T,
x:["One number appears more than half the time. Find it.","[2,2,1,1,1,2,2]: 2 appears 4 of 7 times → 2.","Boyer–Moore voting: keep a candidate and a count. Equal numbers add, different ones cancel out."],
i:"[2,2,1,1,1,2,2]",o:"2",
code:["class Solution:","    def majorityElement(self, nums):","        cand, count = None, 0","        for x in nums:","            if count == 0: cand = x","            count += 1 if x == cand else -1","        return cand"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[2,2,1,1,1,2,2];let c=null,n=0;const f=[F("Start","No candidate yet.",[L("cand, count")],[R("nums",a)],{cand:"None",count:0})];
 a.forEach((x,i)=>{if(n===0)c=x;n+=x===c?1:-1;f.push(F("x = "+x,(x===c?"Same as candidate → count up.":"Different → cancel one vote.")+" Candidate "+c+".",[L("if count == 0"),L("count += 1")],[R("nums",a,[i],M({x:i}))],{cand:c,count:n}))});
 f.push(F("Return","The majority survives all cancelling.",[L("return cand")],[R("answer",[c],[0])]));return f}});
prob({id:189,slug:"rotate-array",title:"Rotate Array",level:"Medium",topic:T,
x:["Rotate the list to the right by k steps, in place.","[1..7], k = 3: the last 3 numbers move to the front → [5,6,7,1,2,3,4].","Reverse everything, then reverse the first k, then reverse the rest."],
i:"[1,2,3,4,5,6,7], k=3",o:"[5,6,7,1,2,3,4]",
code:["class Solution:","    def rotate(self, nums, k):","        k %= len(nums)","        def rev(a, b):","            while a < b:","                nums[a], nums[b] = nums[b], nums[a]; a += 1; b -= 1","        rev(0, len(nums) - 1)","        rev(0, k - 1)","        rev(k, len(nums) - 1)"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","k = 3.",[L("k %=")],[R("nums",[1,2,3,4,5,6,7])]),F("Reverse all","Whole list flipped.",[L("rev(0, len(nums) - 1)")],[R("nums",[7,6,5,4,3,2,1],range(0,6))]),F("Reverse first k","First 3 flipped back: 5,6,7.",[L("rev(0, k - 1)")],[R("nums",[5,6,7,4,3,2,1],[0,1,2])]),F("Reverse the rest","Last 4 flipped back: 1,2,3,4.",[L("rev(k, len(nums) - 1)")],[R("nums",[5,6,7,1,2,3,4],[3,4,5,6])])]});
prob({id:121,slug:"best-time-to-buy-and-sell-stock",title:"Best Time to Buy and Sell Stock",level:"Easy",topic:T,
x:["Prices per day. Buy once, sell once later. What's the biggest profit (0 if none)?","[7,1,5,3,6,4]: buy at 1, sell at 6 → profit 5.","Track the lowest price so far; profit today = price − lowest."],
i:"[7,1,5,3,6,4]",o:"5",
code:["class Solution:","    def maxProfit(self, prices):","        low, best = float('inf'), 0","        for p in prices:","            low = min(low, p)","            best = max(best, p - low)","        return best"],
time:"O(n)",space:"O(1)",
frames:L=>{const p=[7,1,5,3,6,4];let low=Infinity,b=0;const f=[F("Start","Nothing bought yet.",[L("low, best")],[R("prices",p)])];
 p.forEach((x,i)=>{low=Math.min(low,x);b=Math.max(b,x-low);f.push(F("Day "+i+" (price "+x+")","lowest so far "+low+"; profit if sold now "+(x-low),[L("low = min"),L("best = max")],[R("prices",p,[i]),R("low",[low])],{best:b}))});
 f.push(F("Return","Best profit = "+b+".",[L("return best")],[R("answer",[b],[0])]));return f}});
prob({id:122,slug:"best-time-to-buy-and-sell-stock-ii",title:"Best Time to Buy and Sell Stock II",level:"Medium",topic:T,
x:["Prices per day. Buy and sell as many times as you like (hold one share at most). Maximum profit?","[7,1,5,3,6,4]: buy 1 sell 5 (+4), buy 3 sell 6 (+3) → 7.","Grab every upward step: add up each day's gain when the price rose."],
i:"[7,1,5,3,6,4]",o:"7",
code:["class Solution:","    def maxProfit(self, prices):","        profit = 0","        for i in range(1, len(prices)):","            if prices[i] > prices[i - 1]:","                profit += prices[i] - prices[i - 1]","        return profit"],
time:"O(n)",space:"O(1)",
frames:L=>{const p=[7,1,5,3,6,4];let t=0;const f=[F("Start","Collect every price rise.",[L("profit = 0")],[R("prices",p)])];
 for(let i=1;i<p.length;i++){const up=p[i]>p[i-1];if(up)t+=p[i]-p[i-1];f.push(F("Day "+i,up?"Up "+p[i-1]+" → "+p[i]+": +"+(p[i]-p[i-1]):"Down — no trade.",[L("if prices[i]"),...(up?[L("profit +=")]:[])],[R("prices",p,[i-1,i])],{profit:t}))}
 f.push(F("Return","Total profit = "+t+".",[L("return profit")],[R("answer",[t],[0])]));return f}});
prob({id:55,slug:"jump-game",title:"Jump Game",level:"Medium",topic:T,
x:["Each number is the maximum jump length from that spot. Starting at the first spot, can you reach the last?","[2,3,1,1,4]: from 0 you can reach 1 or 2; from 1 you can jump up to 4 → the end is reachable → True.","Track the farthest spot you can reach. If you ever stand beyond it, you're stuck."],
i:"[2,3,1,1,4]",o:"True",
code:["class Solution:","    def canJump(self, nums):","        reach = 0","        for i, x in enumerate(nums):","            if i > reach: return False","            reach = max(reach, i + x)","        return True"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[2,3,1,1,4];let r=0;const f=[F("Start","Farthest reachable = 0.",[L("reach = 0")],[R("nums",a)],{reach:0})];
 a.forEach((x,i)=>{r=Math.max(r,i+x);f.push(F("i = "+i,"From here you can reach up to "+(i+x)+" → farthest = "+r+".",[L("if i > reach"),L("reach = max")],[R("nums",a,range(i,Math.min(r,4)),M({i}))],{reach:r}))});
 f.push(F("Return","Farthest 4 ≥ last index → True.",[L("return True")],[R("nums",a,[4])]));return f}});
prob({id:45,slug:"jump-game-ii",title:"Jump Game II",level:"Medium",topic:T,
x:["Same jump rules, and the end is always reachable. What's the fewest jumps to reach the last spot?","[2,3,1,1,4]: jump 0→1 (1 jump), then 1→4 (2 jumps) → 2.","Think in 'waves': when you reach the end of the current wave, take a jump to the farthest spot seen."],
i:"[2,3,1,1,4]",o:"2",
code:["class Solution:","    def jump(self, nums):","        jumps = cur_end = far = 0","        for i in range(len(nums) - 1):","            far = max(far, i + nums[i])","            if i == cur_end:","                jumps += 1; cur_end = far","        return jumps"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[2,3,1,1,4];let j=0,ce=0,far=0;const f=[F("Start","No jumps yet.",[L("jumps = cur_end")],[R("nums",a,[0],M({end:0}))])];
 for(let i=0;i<4;i++){far=Math.max(far,i+a[i]);let d="farthest = "+far;let jm=false;if(i===ce){j++;ce=far;jm=true;d+=" · end of wave → jump! next wave ends at "+ce}f.push(F("i = "+i,d,[L("far = max"),...(jm?[L("jumps += 1")]:[])],[R("nums",a,[i],M({i,end:ce}))],{jumps:j,far}))}
 f.push(F("Return","2 jumps.",[L("return jumps")],[R("answer",[j],[0])]));return f}});
prob({id:274,slug:"h-index",title:"H-Index",level:"Medium",topic:T,
x:["Citations per paper. The h-index is the biggest h where h papers have at least h citations each.","[3,0,6,1,5] sorted high→low: 6,5,3,1,0. Three papers have ≥ 3 citations → h = 3.","Sort descending, then count while the citation at position h is bigger than h."],
i:"[3,0,6,1,5]",o:"3",
code:["class Solution:","    def hIndex(self, citations):","        citations.sort(reverse=True)","        h = 0","        while h < len(citations) and citations[h] > h:","            h += 1","        return h"],
time:"O(n log n)",space:"O(1)",
frames:L=>{const a=[6,5,3,1,0];const f=[F("Sort","Highest citations first.",[L("citations.sort")],[R("citations",a)])];let h=0;
 while(h<5&&a[h]>h){h++;f.push(F("h → "+h,a[h-1]+" > "+(h-1)+" → at least "+h+" papers have ≥ "+h+" citations.",[L("while h <"),L("h += 1")],[R("citations",a,range(0,h-1))],{h}))}
 f.push(F("Stop",a[h]+" > "+h+" is false.",[L("while h <")],[R("citations",a,[h])],{h}));f.push(F("Return","h-index = 3.",[L("return h")],[R("answer",[h],[0])]));return f}});
prob({id:380,slug:"insert-delete-getrandom-o1",title:"Insert Delete GetRandom O(1)",level:"Medium",topic:T,
x:["Build a set where insert, remove and getRandom all run in O(1) time.","insert 1, insert 2, remove 1, getRandom (only 2 is left).","A list for random access + a dictionary of value → index. To remove, swap the last item into the hole."],
i:"insert(1), insert(2), remove(1), getRandom()",o:"True, True, True, 2",
code:["class RandomizedSet:","    def __init__(self):","        self.vals, self.pos = [], {}","    def insert(self, v):","        if v in self.pos: return False","        self.pos[v] = len(self.vals); self.vals.append(v); return True","    def remove(self, v):","        if v not in self.pos: return False","        i, last = self.pos[v], self.vals[-1]","        self.vals[i] = last; self.pos[last] = i","        self.vals.pop(); del self.pos[v]; return True","    def getRandom(self):","        return random.choice(self.vals)"],
time:"O(1) per op",space:"O(n)",
frames:L=>[F("insert(1)","Append 1; remember its index 0.",[L("self.pos[v] = len")],[R("vals",[1],[0]),R("index of",["1→0"])]),F("insert(2)","Append 2; index 1.",[L("self.pos[v] = len")],[R("vals",[1,2],[1]),R("index of",["1→0","2→1"])]),F("remove(1)","1 sits at index 0. Move the last item (2) into slot 0.",[L("i, last"),L("self.vals[i] = last")],[R("vals",[2,2],[0,1]),R("index of",["2→0"])]),F("Pop","Drop the duplicate tail → list is just [2].",[L("self.vals.pop()")],[R("vals",[2],[0]),R("index of",["2→0"])]),F("getRandom()","Pick a random index of the list → 2.",[L("random.choice")],[R("vals",[2],[0])])]});
prob({id:134,slug:"gas-station",title:"Gas Station",level:"Medium",topic:T,
x:["Gas stations in a circle. gas[i] is fuel you get at station i, cost[i] is fuel to drive to the next one. Which station can you start from to complete a lap? (−1 if none)","gas [1,2,3,4,5], cost [3,4,5,1,2] → start at station 3.","If the tank goes negative, the start can't be anywhere before here — restart from the next station."],
i:"gas=[1,2,3,4,5], cost=[3,4,5,1,2]",o:"3",
code:["class Solution:","    def canCompleteCircuit(self, gas, cost):","        if sum(gas) < sum(cost): return -1","        tank = start = 0","        for i in range(len(gas)):","            tank += gas[i] - cost[i]","            if tank < 0:","                start, tank = i + 1, 0","        return start"],
time:"O(n)",space:"O(1)",
frames:L=>{const g=[1,2,3,4,5],c=[3,4,5,1,2],d=g.map((x,i)=>x-c[i]);let t=0,s=0;const f=[F("Net fuel","gas − cost at each station.",[L("if sum(gas)")],[R("gas − cost",d)],{total:"≥ 0 → possible"})];
 d.forEach((x,i)=>{t+=x;let m="tank = "+t;let rs=false;if(t<0){s=i+1;t=0;rs=true;m="Tank negative → restart from station "+s}f.push(F("Station "+i,m,[L("tank +="),...(rs?[L("start, tank")]:[])],[R("gas − cost",d,[i],M({start:s<5?s:null}))],{tank:t}))});
 f.push(F("Return","Start at station "+s+".",[L("return start")],[R("gas − cost",d,[s],M({start:s}))]));return f}});
prob({id:135,slug:"candy",title:"Candy",level:"Hard",topic:T,
x:["Children in a line with ratings. Everyone gets at least 1 candy, and a child with a higher rating than a neighbour gets more candy than that neighbour. Fewest candies?","ratings [1,0,2] → candies [2,1,2] = 5.","Two passes: left→right fixes 'higher than left neighbour', right→left fixes 'higher than right neighbour'."],
i:"ratings=[1,0,2]",o:"5",
code:["class Solution:","    def candy(self, ratings):","        n = len(ratings); c = [1] * n","        for i in range(1, n):","            if ratings[i] > ratings[i-1]: c[i] = c[i-1] + 1","        for i in range(n - 2, -1, -1):","            if ratings[i] > ratings[i+1]: c[i] = max(c[i], c[i+1] + 1)","        return sum(c)"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Start","Everyone gets 1.",[L("c = [1]")],[R("ratings",[1,0,2]),R("candy",[1,1,1])]),F("Left → right","Child 2 (rating 2) beats the left neighbour → 2 candies.",[L("if ratings[i] > ratings[i-1]")],[R("ratings",[1,0,2],[2]),R("candy",[1,1,2],[2])]),F("Right → left","Child 0 (rating 1) beats the right neighbour (0) → needs more than 1 → 2.",[L("if ratings[i] > ratings[i+1]")],[R("ratings",[1,0,2],[0]),R("candy",[2,1,2],[0])]),F("Return","2 + 1 + 2 = 5.",[L("return sum")],[R("candy",[2,1,2],[0,1,2])],{total:5})]});
prob({id:42,slug:"trapping-rain-water",title:"Trapping Rain Water",level:"Hard",topic:T,
x:["Bars of different heights. After rain, how much water is trapped between the bars?","[0,1,0,2,1,0,1,3,2,1,2,1] traps 6 units.","Water above a bar = (shorter of the tallest bar on each side) − its height. Use two pointers moving inward."],
i:"[0,1,0,2,1,0,1,3,2,1,2,1]",o:"6",
code:["class Solution:","    def trap(self, height):","        l, r = 0, len(height) - 1","        lmax = rmax = water = 0","        while l < r:","            if height[l] < height[r]:","                lmax = max(lmax, height[l]); water += lmax - height[l]; l += 1","            else:","                rmax = max(rmax, height[r]); water += rmax - height[r]; r -= 1","        return water"],
time:"O(n)",space:"O(1)",
frames:L=>{const h=[0,1,0,2,1,0,1,3,2,1,2,1];let l=0,r=11,lm=0,rm=0,w=0;const f=[F("Start","Two pointers at both ends.",[L("l, r =")],[R("height",h,[],M({l:0,r:11}))])];
 while(l<r){let d,ol=l,or=r;if(h[l]<h[r]){lm=Math.max(lm,h[l]);w+=lm-h[l];d="Left is lower: left max "+lm+", water +"+(lm-h[l]);l++}else{rm=Math.max(rm,h[r]);w+=rm-h[r];d="Right is lower: right max "+rm+", water +"+(rm-h[r]);r--}
 f.push(F("l="+ol+", r="+or,d,[L("if height[l] < height[r]"),L(h[ol]<h[or]?"lmax = max":"rmax = max")],[R("height",h,[h[ol]<h[or]?ol:or],M({l:ol,r:or}))],{water:w}))}
 f.push(F("Return","6 units trapped.",[L("return water")],[R("answer",[w],[0])]));return f}});
prob({id:13,slug:"roman-to-integer",title:"Roman to Integer",level:"Easy",topic:T,
x:["Convert a Roman numeral to a number.","MCMXCIV = 1000 + (1000−100) + (100−10) + (5−1) = 1994.","If a symbol is smaller than the next one, subtract it; otherwise add it."],
i:'"MCMXCIV"',o:"1994",
code:["class Solution:","    def romanToInt(self, s):","        v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}","        total = 0","        for i, ch in enumerate(s):","            if i + 1 < len(s) and v[ch] < v[s[i+1]]: total -= v[ch]","            else: total += v[ch]","        return total"],
time:"O(n)",space:"O(1)",
frames:L=>{const s="MCMXCIV",v={I:1,V:5,X:10,L:50,C:100,D:500,M:1000};let t=0;const f=[F("Start","Read left to right.",[L("total = 0")],[R("s",s)])];
 [...s].forEach((c,i)=>{const sub=i+1<s.length&&v[c]<v[s[i+1]];t+=sub?-v[c]:v[c];f.push(F(c+" ("+v[c]+")",sub?c+" is smaller than "+s[i+1]+" → subtract.":"Add.",[L("if i + 1"),L(sub?"total -= v[ch]":"else: total +=")],[R("s",s,[i],M({i}))],{total:t}))});
 f.push(F("Return","1994",[L("return total")],[R("answer",[t],[0])]));return f}});
prob({id:12,slug:"integer-to-roman",title:"Integer to Roman",level:"Medium",topic:T,
x:["Convert a number to a Roman numeral.","1994 = M (1000) + CM (900) + XC (90) + IV (4) → MCMXCIV.","Go through the values from biggest to smallest, taking each as many times as it fits."],
i:"1994",o:'"MCMXCIV"',
code:["class Solution:","    def intToRoman(self, num):","        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]","        res = ''","        for v, s in vals:","            while num >= v:","                res += s; num -= v","        return res"],
time:"O(1)",space:"O(1)",
frames:L=>{const f=[F("Start","Remaining number 1994.",[L("vals =")],[R("remaining",[1994]),R("result",[E])])];let n=1994,res="";
 for(const[v,s]of [[1000,"M"],[900,"CM"],[90,"XC"],[4,"IV"]]){n-=v;res+=s;f.push(F("Take "+s+" ("+v+")","Remaining "+n+".",[L("while num >= v"),L("res += s")],[R("remaining",[n]),R("result",[res],[0])]))}
 f.push(F("Return","MCMXCIV",[L("return res")],[R("answer",res,range(0,res.length-1))]));return f}});
prob({id:58,slug:"length-of-last-word",title:"Length of Last Word",level:"Easy",topic:T,
x:["A sentence with spaces. How long is the last word?","'   fly me   to   the moon  ' → last word is 'moon' → 4.","split() throws away extra spaces; take the last piece."],
i:'"   fly me   to   the moon  "',o:"4",
code:["class Solution:","    def lengthOfLastWord(self, s):","        return len(s.split()[-1])"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Input","Spaces at the start, between and at the end.",[L("def length")],[R("s","   fly me   to   the moon  ".replace(/ /g," "))]),F("split()","Words only: fly, me, to, the, moon.",[L("s.split()")],[R("words",["fly","me","to","the","moon"],[4])]),F("Last word","[-1] = 'moon'.",[L("[-1]")],[R("last","moon",[0,1,2,3])]),F("Length","len('moon') = 4",[L("len(")],[R("answer",[4],[0])])]});
prob({id:14,slug:"longest-common-prefix",title:"Longest Common Prefix",level:"Easy",topic:T,
x:["Find the longest beginning shared by all the words.","flower, flow, flight all start with 'fl'.","Start with the first word as the prefix and shorten it until each other word starts with it."],
i:'["flower","flow","flight"]',o:'"fl"',
code:["class Solution:","    def longestCommonPrefix(self, strs):","        prefix = strs[0]","        for s in strs[1:]:","            while not s.startswith(prefix):","                prefix = prefix[:-1]","        return prefix"],
time:"O(total chars)",space:"O(1)",
frames:L=>[F("Start","prefix = 'flower'.",[L("prefix = strs[0]")],[R("prefix","flower",range(0,5))]),F("Check 'flow'","'flow' doesn't start with 'flower' → trim to 'flowe', 'flow'.",[L("while not s.startswith"),L("prefix = prefix[:-1]")],[R("prefix","flow",range(0,3))]),F("Check 'flight'","'flight' doesn't start with 'flow', 'flo' → trim to 'fl'.",[L("while not s.startswith"),L("prefix = prefix[:-1]")],[R("prefix","fl",[0,1])]),F("Return","'fl'",[L("return prefix")],[R("answer","fl",[0,1])])]});
prob({id:6,slug:"zigzag-conversion",title:"Zigzag Conversion",level:"Medium",topic:T,
x:["Write the text in a zigzag down and up over numRows rows, then read it row by row.","PAYPALISHIRING on 3 rows reads PAHNAPLSIIGYIR.","Walk through the letters, moving the row down then up, bouncing at the top and bottom."],
i:'"PAYPALISHIRING", numRows=3',o:'"PAHNAPLSIIGYIR"',
code:["class Solution:","    def convert(self, s, numRows):","        if numRows == 1: return s","        rows, r, d = [''] * numRows, 0, 1","        for ch in s:","            rows[r] += ch","            if r == 0: d = 1","            elif r == numRows - 1: d = -1","            r += d","        return ''.join(rows)"],
time:"O(n)",space:"O(n)",
frames:L=>{const s="PAYPALISHIRING",rows=["","",""];let r=0,d=1;const f=[F("Start","3 empty rows; direction starts down.",[L("rows, r, d")],[R("row 0",[E]),R("row 1",[E]),R("row 2",[E])])];
 [...s].forEach((c,i)=>{rows[r]+=c;const used=r;if(r===0)d=1;else if(r===2)d=-1;r+=d;f.push(F("'"+c+"'","Place in row "+used+".",[L("rows[r] += ch"),L("r += d")],rows.map((x,k)=>R("row "+k,x?x:[E],k===used&&x?[x.length-1]:[]))))});
 f.push(F("Return","Join the rows.",[L("return ''.join")],[R("answer","PAHNAPLSIIGYIR",[])]));return f}});
prob({id:28,slug:"find-the-index-of-the-first-occurrence",title:"Find the Index of the First Occurrence in a String",level:"Easy",topic:T,
x:["Find where the word needle first appears inside haystack. Return −1 if it doesn't.","haystack 'hello', needle 'll' → starts at index 2.","Slide a window the size of the needle along the haystack and compare."],
i:'haystack="hello", needle="ll"',o:"2",
code:["class Solution:","    def strStr(self, haystack, needle):","        for i in range(len(haystack) - len(needle) + 1):","            if haystack[i:i+len(needle)] == needle:","                return i","        return -1"],
time:"O(n·m)",space:"O(1)",
frames:L=>{const h="hello",f=[F("Start","Compare each 2-letter window with 'll'.",[L("for i in range")],[R("haystack",h)])];
 for(let i=0;i<4;i++){const w=h.slice(i,i+2),ok=w==="ll";f.push(F("i = "+i,"'"+w+"' "+(ok?"== 'll' → found!":"≠ 'll'."),[L("if haystack[i:"),...(ok?[L("return i")]:[])],[R("haystack",h,[i,i+1],M({i}))]));if(ok)break}return f}});
prob({id:68,slug:"text-justification",title:"Text Justification",level:"Hard",topic:T,
x:["Pack words into lines of exactly maxWidth characters, spreading extra spaces evenly between words. The last line is left-aligned.","Words This, is, an | example, of, text | justification. with width 16.","Fit as many words as possible per line; then share leftover spaces across the gaps (extras go to the left gaps)."],
i:'["This","is","an","example","of","text","justification."], maxWidth=16',o:'["This    is    an","example  of text","justification.  "]',
code:["class Solution:","    def fullJustify(self, words, maxWidth):","        res, line, length = [], [], 0","        for w in words:","            if length + len(w) + len(line) > maxWidth:","                gaps = len(line) - 1 or 1","                for i in range(maxWidth - length):","                    line[i % gaps] += ' '","                res.append(''.join(line)); line, length = [], 0","            line.append(w); length += len(w)","        res.append(' '.join(line).ljust(maxWidth))","        return res"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Fill line 1","'This', 'is', 'an' fit (8 letters). Adding 'example' would overflow.",[L("if length + len(w)")],[R("line",["This","is","an"],[0,1,2]),R("letters",[8])]),F("Spread spaces","16 − 8 = 8 spaces over 2 gaps → 4 each.",[L("gaps ="),L("line[i % gaps]")],[R("result",["This    is    an"],[0])]),F("Fill line 2","'example','of','text' = 13 letters + 2 gaps = 15. 'justification.' won't fit.",[L("if length + len(w)")],[R("line",["example","of","text"],[0,1,2])]),F("Spread spaces","3 spaces over 2 gaps → extras go left: 2 then 1.",[L("line[i % gaps]")],[R("result",["This    is    an","example  of text"],[1])]),F("Last line","Left-align, pad the right.",[L("ljust")],[R("result",["This    is    an","example  of text","justification.  "],[2])])]});
