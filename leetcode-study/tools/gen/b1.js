const {prob,R,M,F,range}=require("./lib");
// 1768
prob({id:1768,slug:"merge-strings-alternately",title:"Merge Strings Alternately",level:"Easy",topic:"Array / String",
q:"Merge two strings by taking letters alternately; append any leftover letters.",i:'"abc", "pq"',o:'"apbqc"',
code:["class Solution:","    def mergeAlternately(self, word1, word2):","        res = []","        for i in range(max(len(word1), len(word2))):","            if i < len(word1): res.append(word1[i])","            if i < len(word2): res.append(word2[i])","        return ''.join(res)"],
time:"O(n+m)",space:"O(n+m)",explain:"One loop up to the longer length, guarding each side.",best:"One pass with index guards handles unequal lengths with no special case.",
notes:["Loop to max(len1, len2); guard each append with i < len.","Building a list and joining once beats repeated string +=."],
frames:L=>{const a="abc",b="pq",f=[F("Input","Two words, different lengths.",[L("def merge")],[R("word1",a),R("word2",b)])];let res=[];
 for(let i=0;i<3;i++){const t=[];if(i<a.length){res.push(a[i]);t.push("word1["+i+"]")}if(i<b.length){res.push(b[i]);t.push("word2["+i+"]")}
 f.push(F("i = "+i,"Append "+t.join(" then ")+".",[L("for i"),L("if i < len(word1)"),L("if i < len(word2)")],[R("word1",a,[i],M({i})),R("word2",b,i<b.length?[i]:[],M({i})),R("res",res,[res.length-1])]))}
 f.push(F("Join","Join the list into the answer.",[L("return")],[R("answer","apbqc",[3,4])]));return f}});
// 443
prob({id:443,slug:"string-compression",title:"String Compression",level:"Medium",topic:"Array / String",
q:"Compress runs in place: letter, then the count if it is greater than 1. Return new length.",i:'["a","a","b","b","c","c","c"]',o:'6 → ["a","2","b","2","c","3"]',
code:["class Solution:","    def compress(self, chars):","        write = read = 0","        while read < len(chars):","            ch, count = chars[read], 0","            while read < len(chars) and chars[read] == ch:","                read += 1","                count += 1","            chars[write] = ch","            write += 1","            if count > 1:","                for d in str(count):","                    chars[write] = d","                    write += 1","        return write"],
time:"O(n)",space:"O(1)",explain:"Read pointer scans a run; write pointer lays down the compressed form.",best:"Read/write pointers compress in place with O(1) extra space.",
notes:["write never passes read, so overwriting is safe.","Counts above 9 are written digit by digit."],
frames:L=>{const c=["a","a","b","b","c","c","c"],f=[F("Input","Runs: aa, bb, ccc.",[L("write = read")],[R("chars",c,[],M({read:0,write:0}))])];
 let w=0,r=0;const out=c.slice();
 while(r<c.length){const ch=c[r],s=r;let n=0;while(r<c.length&&c[r]===ch){r++;n++}
  f.push(F("Run of "+ch,"Read "+n+" × '"+ch+"'.",[L("ch, count"),L("while read < len(chars) and")],[R("chars",out,range(s,r-1),M({read:r,write:w}))],{count:n}));
  out[w++]=ch;if(n>1)for(const d of String(n))out[w++]=d;
  f.push(F("Write "+ch+(n>1?n:""),"Write letter"+(n>1?" and count":"")+" at the write pointer.",[L("chars[write] = ch"),L("if count > 1")],[R("chars",out,range(w-(n>1?2:1),w-1),M({read:r,write:w}))]));}
 f.push(F("Return","Answer is the write position.",[L("return write")],[R("chars",out.slice(0,w),range(0,w-1))],{length:w}));return f}});
// 283
prob({id:283,slug:"move-zeroes",title:"Move Zeroes",level:"Easy",topic:"Two Pointers",
q:"Move all zeroes to the end in place, keeping the order of non-zero values.",i:"[0, 1, 0, 3, 12]",o:"[1, 3, 12, 0, 0]",
code:["class Solution:","    def moveZeroes(self, nums):","        slow = 0","        for fast in range(len(nums)):","            if nums[fast] != 0:","                nums[slow], nums[fast] = nums[fast], nums[slow]","                slow += 1"],
time:"O(n)",space:"O(1)",explain:"slow marks where the next non-zero belongs.",best:"A single pass with a slow write pointer; no extra array.",
notes:["slow always points at the first zero (or the next free slot).","Swapping keeps zeros moving right."],
frames:L=>{const a=[0,1,0,3,12],f=[F("Input","slow = 0; fast scans.",[L("slow = 0")],[R("nums",a,[],M({slow:0}))])];let s=0;
 for(let fa=0;fa<a.length;fa++){if(a[fa]!==0){[a[s],a[fa]]=[a[fa],a[s]];s++;f.push(F("fast = "+fa,"Non-zero: swap into slow, then slow++.",[L("if nums[fast]"),L("nums[slow], nums[fast]"),L("slow += 1")],[R("nums",a,[s-1,fa],M({slow:s,fast:fa}))]))}
 else f.push(F("fast = "+fa,"Zero: skip.",[L("if nums[fast]")],[R("nums",a,[fa],M({slow:s,fast:fa}))]))}return f}});
// 392
prob({id:392,slug:"is-subsequence",title:"Is Subsequence",level:"Easy",topic:"Two Pointers",
q:"Is s a subsequence of t (same order, gaps allowed)?",i:'s="ace", t="abcde"',o:"True",
code:["class Solution:","    def isSubsequence(self, s, t):","        i = 0","        for ch in t:","            if i < len(s) and ch == s[i]:","                i += 1","        return i == len(s)"],
time:"O(n)",space:"O(1)",explain:"Advance in s only when t's letter matches.",best:"One scan of t with a pointer into s.",
notes:["Never move backwards in either string.","Match greedily: the earliest match is always safe."],
frames:L=>{const s="ace",t="abcde",f=[F("Input","Find a, c, e in order inside t.",[L("i = 0")],[R("s",s,[],M({i:0})),R("t",t)])];let i=0;
 for(let k=0;k<t.length;k++){const m=i<s.length&&t[k]===s[i];if(m)i++;f.push(F("t["+k+"] = "+t[k],m?"Match! Move i to "+i+".":"No match, keep i.",[L("if i < len(s)"),...(m?[L("i += 1")]:[])],[R("s",s,m?[i-1]:[],M({i})),R("t",t,m?[k]:[],M({ch:k}))]))}
 f.push(F("Return","i reached len(s) = 3.",[L("return")],[R("s",s,[0,1,2])],{result:"True"}));return f}});
// 11
prob({id:11,slug:"container-with-most-water",title:"Container With Most Water",level:"Medium",topic:"Two Pointers",
q:"Pick two lines that, with the x-axis, hold the most water.",i:"[1, 8, 6, 2, 5, 4, 8, 3, 7]",o:"49",
code:["class Solution:","    def maxArea(self, height):","        l, r = 0, len(height) - 1","        best = 0","        while l < r:","            best = max(best, min(height[l], height[r]) * (r - l))","            if height[l] < height[r]: l += 1","            else: r -= 1","        return best"],
time:"O(n)",space:"O(1)",explain:"Always move the shorter wall inward.",best:"Two pointers: moving the taller wall can never help, so move the shorter one.",
notes:["Area = min(height) × width.","Width shrinks every step, so only a taller short-wall can improve."],
frames:L=>{const h=[1,8,6,2,5,4,8,3,7],f=[F("Input","Start with the widest container.",[L("l, r =")],[R("height",h,[0,8],M({l:0,r:8}))])];let l=0,r=8,b=0;
 while(l<r){const a=Math.min(h[l],h[r])*(r-l);b=Math.max(b,a);const ol=l,or=r;if(h[l]<h[r])l++;else r--;
 f.push(F("l="+ol+", r="+or,"area = "+Math.min(h[ol],h[or])+" × "+(or-ol)+" = "+a+". Move the shorter wall.",[L("best = max"),L("if height[l]")],[R("height",h,[ol,or],M({l:ol,r:or}))],{area:a,best:b}))}
 f.push(F("Return","Pointers met.",[L("return best")],[R("height",h,[1,8])],{best:b}));return f}});
// 1679
prob({id:1679,slug:"max-number-of-k-sum-pairs",title:"Max Number of K-Sum Pairs",level:"Medium",topic:"Two Pointers",
q:"Repeatedly remove two numbers that add to k. How many operations?",i:"[3, 1, 3, 4, 3], k=6",o:"1",
code:["class Solution:","    def maxOperations(self, nums, k):","        nums.sort()","        l, r, ops = 0, len(nums) - 1, 0","        while l < r:","            s = nums[l] + nums[r]","            if s == k:","                ops += 1; l += 1; r -= 1","            elif s < k: l += 1","            else: r -= 1","        return ops"],
time:"O(n log n)",space:"O(1)",explain:"Sort, then squeeze from both ends.",best:"Sort + two pointers; a hash-map count gives O(n) if order doesn't matter.",
notes:["Sum too small → move left up; too big → move right down.","Each number is used at most once."],
frames:L=>{const a=[1,3,3,3,4],k=6,f=[F("Sorted","Sort first: [1,3,3,3,4], k = 6.",[L("nums.sort")],[R("nums",a,[],M({l:0,r:4}))])];let l=0,r=4,o=0;
 while(l<r){const s=a[l]+a[r],ol=l,or=r;let d;if(s===k){o++;l++;r--;d="= k → pair! ops = "+o}else if(s<k){l++;d="< k → move l"}else{r--;d="> k → move r"}
 f.push(F("sum = "+s,a[ol]+" + "+a[or]+" "+d+".",[L("s = nums"),L(s===k?"ops += 1":s<k?"elif s < k":"else: r")],[R("nums",a,[ol,or],M({l:ol,r:or}))],{ops:o}))}
 f.push(F("Return","l met r.",[L("return ops")],[R("nums",a)],{ops:o}));return f}});
// 643
prob({id:643,slug:"maximum-average-subarray-i",title:"Maximum Average Subarray I",level:"Easy",topic:"Sliding Window",
q:"Find the contiguous subarray of length k with the largest average.",i:"[1, 12, -5, -6, 50, 3], k=4",o:"12.75",
code:["class Solution:","    def findMaxAverage(self, nums, k):","        window = sum(nums[:k])","        best = window","        for i in range(k, len(nums)):","            window += nums[i] - nums[i - k]","            best = max(best, window)","        return best / k"],
time:"O(n)",space:"O(1)",explain:"Slide the window: add the new element, drop the old one.",best:"Sliding window updates the sum in O(1) per step.",
notes:["Compare sums, divide once at the end.","Never recompute the whole window."],
frames:L=>{const a=[1,12,-5,-6,50,3],k=4;let w=a.slice(0,k).reduce((x,y)=>x+y),b=w;const f=[F("First window","Sum of the first 4 = "+w+".",[L("window = sum")],[R("nums",a,range(0,k-1))],{window:w,best:b})];
 for(let i=k;i<a.length;i++){w+=a[i]-a[i-k];b=Math.max(b,w);f.push(F("Slide to i="+i,"Add "+a[i]+", drop "+a[i-k]+".",[L("window +="),L("best = max")],[R("nums",a,range(i-k+1,i))],{window:w,best:b}))}
 f.push(F("Return",b+" / "+k+" = "+(b/k),[L("return")],[R("nums",a,range(1,4))],{answer:b/k}));return f}});
// 1456
prob({id:1456,slug:"max-vowels-in-substring",title:"Maximum Vowels in a Substring of Given Length",level:"Medium",topic:"Sliding Window",
q:"Most vowels in any substring of length k?",i:'"abciiidef", k=3',o:"3",
code:["class Solution:","    def maxVowels(self, s, k):","        vowels = set('aeiou')","        cnt = sum(c in vowels for c in s[:k])","        best = cnt","        for i in range(k, len(s)):","            cnt += (s[i] in vowels) - (s[i-k] in vowels)","            best = max(best, cnt)","        return best"],
time:"O(n)",space:"O(1)",explain:"Fixed-size window; only the entering and leaving letters matter.",best:"Fixed window: adjust the count by what enters and leaves.",
notes:["Booleans add as 0/1 in Python.","Same shape as every fixed-size window problem."],
frames:L=>{const s="abciiidef",k=3,V="aeiou";let c=[...s.slice(0,k)].filter(x=>V.includes(x)).length,b=c;const f=[F("First window","'abc' has "+c+" vowel.",[L("cnt = sum")],[R("s",s,range(0,k-1))],{cnt:c,best:b})];
 for(let i=k;i<s.length;i++){c+=(V.includes(s[i])?1:0)-(V.includes(s[i-k])?1:0);b=Math.max(b,c);f.push(F("Slide to i="+i,"In: '"+s[i]+"', out: '"+s[i-k]+"'.",[L("cnt +="),L("best = max")],[R("s",s,range(i-k+1,i))],{cnt:c,best:b}))}
 f.push(F("Return","Best window had "+b+" vowels.",[L("return")],[R("s",s,[3,4,5])],{best:b}));return f}});
// 1004
prob({id:1004,slug:"max-consecutive-ones-iii",title:"Max Consecutive Ones III",level:"Medium",topic:"Sliding Window",
q:"Longest run of 1s if you may flip at most k zeros.",i:"[1,0,1,1,0,1], k=1",o:"4",
code:["class Solution:","    def longestOnes(self, nums, k):","        left = zeros = best = 0","        for right, x in enumerate(nums):","            if x == 0: zeros += 1","            while zeros > k:","                if nums[left] == 0: zeros -= 1","                left += 1","            best = max(best, right - left + 1)","        return best"],
time:"O(n)",space:"O(1)",explain:"Grow right; shrink left while too many zeros.",best:"Variable window: expand always, shrink only when invalid.",
notes:["The window never holds more than k zeros.","left only moves forward, so it is O(n) total."],
frames:L=>{const a=[1,0,1,1,0,1],k=1;let l=0,z=0,b=0;const f=[F("Input","Allow at most "+k+" zero in the window.",[L("left = zeros")],[R("nums",a,[],M({l:0,r:0}))])];
 for(let r=0;r<a.length;r++){if(a[r]===0)z++;let sh=false;while(z>k){if(a[l]===0)z--;l++;sh=true}b=Math.max(b,r-l+1);
 f.push(F("right = "+r,sh?"Too many zeros → shrink left to "+l+".":"Window is valid.",[L("if x == 0"),...(sh?[L("while zeros > k")]:[]),L("best = max")],[R("nums",a,range(l,r),M({l,r}))],{zeros:z,best:b}))}
 f.push(F("Return","Longest valid window = "+b+".",[L("return")],[R("nums",a,[2,3,4,5])],{best:b}));return f}});
// 1493
prob({id:1493,slug:"longest-subarray-after-deleting-one",title:"Longest Subarray of 1's After Deleting One Element",level:"Medium",topic:"Sliding Window",
q:"Delete exactly one element; longest run of 1s left?",i:"[0,1,1,1,0,1,1,0,1]",o:"5",
code:["class Solution:","    def longestSubarray(self, nums):","        left = zeros = best = 0","        for right, x in enumerate(nums):","            if x == 0: zeros += 1","            while zeros > 1:","                if nums[left] == 0: zeros -= 1","                left += 1","            best = max(best, right - left)","        return best"],
time:"O(n)",space:"O(1)",explain:"Window with at most one zero; the deleted slot costs 1 (right - left).",best:"Same window as 1004 with k = 1, minus the deleted element.",
notes:["right - left (not +1) accounts for the deleted element.","You must delete one even if all are 1s."],
frames:L=>{const a=[0,1,1,1,0,1,1,0,1];let l=0,z=0,b=0;const f=[F("Input","Window may hold at most one zero (the one we delete).",[L("left = zeros")],[R("nums",a,[],M({l:0,r:0}))])];
 for(let r=0;r<a.length;r++){if(a[r]===0)z++;let sh=false;while(z>1){if(a[l]===0)z--;l++;sh=true}b=Math.max(b,r-l);
 f.push(F("right = "+r,sh?"Two zeros → shrink left to "+l+".":"Valid window.",[L("if x == 0"),...(sh?[L("while zeros > 1")]:[]),L("best = max")],[R("nums",a,range(l,r),M({l,r}))],{zeros:z,best:b}))}
 f.push(F("Return","Best = "+b+".",[L("return")],[R("nums",a,[3,4,5,6,7])],{best:b}));return f}});
// 1732
prob({id:1732,slug:"find-the-highest-altitude",title:"Find the Highest Altitude",level:"Easy",topic:"Prefix Sum",
q:"A biker starts at 0 and gains/loses altitude per leg. Highest point?",i:"gain = [-5, 1, 5, 0, -7]",o:"1",
code:["class Solution:","    def largestAltitude(self, gain):","        alt = best = 0","        for g in gain:","            alt += g","            best = max(best, alt)","        return best"],
time:"O(n)",space:"O(1)",explain:"Running sum of the gains; track the max.",best:"A running prefix sum with a max — no array needed.",
notes:["Include the starting altitude 0 in the max.","Prefix sum = running total."],
frames:L=>{const g=[-5,1,5,0,-7];let a=0,b=0;const f=[F("Start","Altitude 0 at the start.",[L("alt = best")],[R("gain",g)],{alt:0,best:0})];
 g.forEach((x,i)=>{a+=x;b=Math.max(b,a);f.push(F("Leg "+(i+1),"alt = "+a+(a===b?" (new best)":""),[L("alt += g"),L("best = max")],[R("gain",g,[i]),R("altitude",[0,...g.slice(0,i+1).map((_,j)=>g.slice(0,j+1).reduce((p,q)=>p+q))],[i+1])],{alt:a,best:b}))});
 f.push(F("Return","Highest point reached.",[L("return")],[R("best",[b],[0])]));return f}});
// 724
prob({id:724,slug:"find-pivot-index",title:"Find Pivot Index",level:"Easy",topic:"Prefix Sum",
q:"Find the index where the sum on the left equals the sum on the right.",i:"[1, 7, 3, 6, 5, 6]",o:"3",
code:["class Solution:","    def pivotIndex(self, nums):","        total, left = sum(nums), 0","        for i, x in enumerate(nums):","            if left == total - left - x: return i","            left += x","        return -1"],
time:"O(n)",space:"O(1)",explain:"right = total − left − x.",best:"Total once, then one pass with a running left sum.",
notes:["No right-sum array: derive it from total.","Return the leftmost pivot."],
frames:L=>{const a=[1,7,3,6,5,6],t=28;let l=0;const f=[F("Total","total = 28, left = 0.",[L("total, left")],[R("nums",a)],{total:t,left:0})];
 for(let i=0;i<a.length;i++){const r=t-l-a[i],ok=l===r;f.push(F("i = "+i,"left = "+l+", right = "+r+(ok?" → equal! pivot":" → not equal"),[L("if left =="),...(ok?[]:[L("left += x")])],[R("nums",a,range(0,i-1).concat(),M({i})),R("right side",a.slice(i+1),[])],{left:l,right:r}));if(ok)break;l+=a[i]}
 f.push(F("Return","Pivot index = 3.",[L("return i")],[R("nums",a,[3],M({pivot:3}))]));return f}});
// 2215
prob({id:2215,slug:"find-difference-of-two-arrays",title:"Find the Difference of Two Arrays",level:"Easy",topic:"Hash Map / Set",
q:"Values only in nums1, and values only in nums2 (no duplicates).",i:"[1,2,3,3], [1,1,2,2]",o:"[[3], []]",
code:["class Solution:","    def findDifference(self, nums1, nums2):","        a, b = set(nums1), set(nums2)","        return [list(a - b), list(b - a)]"],
time:"O(n+m)",space:"O(n+m)",explain:"Sets remove duplicates; subtraction gives differences.",best:"Two sets and set difference.",
notes:["set(x) removes duplicates in O(n).","a - b keeps what a has and b lacks."],
frames:L=>[F("Input","Two arrays with duplicates.",[L("def find")],[R("nums1",[1,2,3,3]),R("nums2",[1,1,2,2])]),
F("To sets","Duplicates vanish.",[L("a, b =")],[R("a",[1,2,3]),R("b",[1,2])]),
F("a − b","Only 3 is in a but not b.",[L("return")],[R("a",[1,2,3],[2]),R("b",[1,2]),R("a − b",[3],[0])]),
F("b − a","Nothing in b is missing from a.",[L("return")],[R("a",[1,2,3]),R("b",[1,2]),R("b − a",["·"],[])]),
F("Return","[[3], []]",[L("return")],[R("answer",["[3]","[]"],[0,1])])]});
// 1207
prob({id:1207,slug:"unique-number-of-occurrences",title:"Unique Number of Occurrences",level:"Easy",topic:"Hash Map / Set",
q:"Is every value's occurrence count different?",i:"[1,2,2,1,1,3]",o:"True",
code:["class Solution:","    def uniqueOccurrences(self, arr):","        count = Counter(arr)","        return len(set(count.values())) == len(count)"],
time:"O(n)",space:"O(n)",explain:"Count, then check the counts are all distinct.",best:"Counter + set of counts; compare sizes.",
notes:["If counts are unique, the set keeps the same size.","Counter is a dict of value → frequency."],
frames:L=>[F("Input","Count each value.",[L("def unique")],[R("arr",[1,2,2,1,1,3])]),
F("Counter","1→3, 2→2, 3→1",[L("count =")],[R("value",[1,2,3]),R("count",[3,2,1],[0,1,2])]),
F("Set of counts","{3,2,1} has 3 items, same as the 3 values.",[L("return")],[R("counts",[3,2,1],[0,1,2]),R("set",[3,2,1],[0,1,2])]),
F("Return","All counts are different.",[L("return")],[R("answer",["True"],[0])])]});
// 1657
prob({id:1657,slug:"determine-if-two-strings-are-close",title:"Determine if Two Strings Are Close",level:"Medium",topic:"Hash Map / Set",
q:"Can you turn one string into the other by swapping letters or swapping letter identities?",i:'"cabbba", "abbccc"',o:"True",
code:["class Solution:","    def closeStrings(self, w1, w2):","        c1, c2 = Counter(w1), Counter(w2)","        return set(c1) == set(c2) and sorted(c1.values()) == sorted(c2.values())"],
time:"O(n log n)",space:"O(1)",explain:"Same letters, and same multiset of frequencies.",best:"Two checks: same letter set, same sorted frequency list.",
notes:["Swapping positions = order is free.","Swapping identities = only the frequency multiset matters."],
frames:L=>[F("Input","Two words.",[L("def close")],[R("w1","cabbba"),R("w2","abbccc")]),
F("Counts","w1: a2 b3 c1 · w2: a1 b2 c3",[L("c1, c2")],[R("w1 letters",["a:2","b:3","c:1"]),R("w2 letters",["a:1","b:2","c:3"])]),
F("Same letters?","Both use {a, b, c}.",[L("set(c1)")],[R("set(c1)",["a","b","c"],[0,1,2]),R("set(c2)",["a","b","c"],[0,1,2])]),
F("Same frequencies?","Sorted: [1,2,3] vs [1,2,3].",[L("sorted(c1")],[R("sorted c1",[1,2,3],[0,1,2]),R("sorted c2",[1,2,3],[0,1,2])]),
F("Return","Both checks pass.",[L("return")],[R("answer",["True"],[0])])]});
// 2352
prob({id:2352,slug:"equal-row-and-column-pairs",title:"Equal Row and Column Pairs",level:"Medium",topic:"Hash Map / Set",
q:"Count (row, column) pairs that contain the same sequence.",i:"[[3,2,1],[1,7,6],[2,7,7]]",o:"1",
code:["class Solution:","    def equalPairs(self, grid):","        rows = Counter(tuple(r) for r in grid)","        cols = zip(*grid)","        return sum(rows[c] for c in cols)"],
time:"O(n²)",space:"O(n²)",explain:"Count rows as tuples, then look up each column.",best:"Hash rows as tuples; each column is one O(n) lookup.",
notes:["Tuples are hashable; lists are not.","zip(*grid) transposes the grid."],
frames:L=>{const g=[[3,2,1],[1,7,6],[2,7,7]];return [F("Grid","Compare every row with every column.",[L("def equal")],[R("grid",g,[],{},"grid")]),
F("Hash rows","Rows become tuples with counts.",[L("rows =")],[R("grid",g,["2,0","2,1","2,2"],{},"grid")]),
F("Columns","zip(*grid) gives columns: (3,1,2) (2,7,7) (1,6,7).",[L("cols =")],[R("grid",g,["0,1","1,1","2,1"],{},"grid")]),
F("Match","Column 1 = (2,7,7) equals row 2 → count 1.",[L("return sum")],[R("row 2",[2,7,7],[0,1,2]),R("col 1",[2,7,7],[0,1,2])]),
F("Return","1 pair.",[L("return")],[R("answer",[1],[0])])]}});
