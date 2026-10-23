const {prob,R,M,F,range}=require("./lib");const E="·";
const GR=(n,g,h=[])=>R(n,g.map(r=>r.slice()),h,{},"grid");
const BSX="Binary Search";
prob({id:35,slug:"search-insert-position",title:"Search Insert Position",level:"Easy",topic:BSX,
x:["In a sorted list, find the index of the target, or the index where it would be inserted to keep the list sorted.","[1,3,5,6], target 2 → it belongs at index 1.","Binary search for the first position whose value is ≥ target."],
i:"[1,3,5,6], target=2",o:"1",
code:["class Solution:","    def searchInsert(self, nums, target):","        lo, hi = 0, len(nums)","        while lo < hi:","            mid = (lo + hi) // 2","            if nums[mid] < target: lo = mid + 1","            else: hi = mid","        return lo"],
time:"O(log n)",space:"O(1)",
frames:L=>{const a=[1,3,5,6],t=2,f=[F("Start","Search range is the whole list.",[L("lo, hi")],[R("nums",a,[],M({lo:0}))])];let lo=0,hi=4;
 while(lo<hi){const mid=(lo+hi)>>1;const d=a[mid]<t?a[mid]+" < 2 → lo = "+(mid+1):a[mid]+" ≥ 2 → hi = "+mid;f.push(F("mid = "+mid,d,[L("mid ="),L(a[mid]<t?"if nums[mid] < target":"else: hi = mid")],[R("nums",a,[mid],M({lo:lo<4?lo:null,mid}))]));if(a[mid]<t)lo=mid+1;else hi=mid}
 f.push(F("Return","Insert at index "+lo+".",[L("return lo")],[R("nums",a,[],M({insert:lo}))]));return f}});
prob({id:74,slug:"search-a-2d-matrix",title:"Search a 2D Matrix",level:"Medium",topic:BSX,
x:["A grid where each row is sorted and each row starts bigger than the previous row ends. Is the target in it?","Target 3 is in the first row → True.","Pretend the grid is one long sorted list and binary search it (index → row = i ÷ n, column = i mod n)."],
i:"[[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=3",o:"True",
code:["class Solution:","    def searchMatrix(self, matrix, target):","        m, n = len(matrix), len(matrix[0])","        lo, hi = 0, m * n - 1","        while lo <= hi:","            mid = (lo + hi) // 2","            v = matrix[mid // n][mid % n]","            if v == target: return True","            if v < target: lo = mid + 1","            else: hi = mid - 1","        return False"],
time:"O(log mn)",space:"O(1)",
frames:L=>{const g=[[1,3,5,7],[10,11,16,20],[23,30,34,60]];return [F("Grid","Treat it like one list of 12 numbers.",[L("lo, hi")],[GR("matrix",g)]),F("mid = 5","Value 11 > 3 → search the left part.",[L("v = matrix"),L("else: hi")],[GR("matrix",g,["1,1"])],{lo:0,hi:11}),F("mid = 2","Value 5 > 3 → left again.",[L("else: hi")],[GR("matrix",g,["0,2"])],{lo:0,hi:4}),F("mid = 0","Value 1 < 3 → right.",[L("if v < target")],[GR("matrix",g,["0,0"])],{lo:0,hi:1}),F("mid = 1","Value 3 = target → True.",[L("if v == target")],[GR("matrix",g,["0,1"])],{lo:1,hi:1})]}});
prob({id:33,slug:"search-in-rotated-sorted-array",title:"Search in Rotated Sorted Array",level:"Medium",topic:BSX,
x:["A sorted list was rotated (e.g. [0,1,2,4,5,6,7] became [4,5,6,7,0,1,2]). Find the target's index in O(log n).","target 0 in [4,5,6,7,0,1,2] → index 4.","One half is always sorted. Check whether the target lies in the sorted half; if yes search there, else the other half."],
i:"[4,5,6,7,0,1,2], target=0",o:"4",
code:["class Solution:","    def search(self, nums, target):","        lo, hi = 0, len(nums) - 1","        while lo <= hi:","            mid = (lo + hi) // 2","            if nums[mid] == target: return mid","            if nums[lo] <= nums[mid]:","                if nums[lo] <= target < nums[mid]: hi = mid - 1","                else: lo = mid + 1","            else:","                if nums[mid] < target <= nums[hi]: lo = mid + 1","                else: hi = mid - 1","        return -1"],
time:"O(log n)",space:"O(1)",
frames:L=>{const a=[4,5,6,7,0,1,2],t=0;let lo=0,hi=6;const f=[F("Start","Whole array.",[L("lo, hi")],[R("nums",a,[],M({lo:0,hi:6}))])];
 while(lo<=hi){const mid=(lo+hi)>>1;if(a[mid]===t){f.push(F("mid = "+mid,"Found 0 at index "+mid+".",[L("if nums[mid] == target")],[R("nums",a,[mid],M({lo,hi,mid}))]));break}let d,ln;
 if(a[lo]<=a[mid]){if(a[lo]<=t&&t<a[mid]){hi=mid-1;d="Left half is sorted and holds the target."}else{lo=mid+1;d="Left half is sorted but target isn't in it → go right."}ln=L("if nums[lo] <= nums[mid]")}else{if(a[mid]<t&&t<=a[hi]){lo=mid+1;d="Right half sorted and holds target."}else{hi=mid-1;d="Right half is sorted but target isn't in it → go left."}ln=L("if nums[mid] < target <= nums[hi]")}
 f.push(F("mid = "+mid,"nums[mid] = "+a[mid]+". "+d,[ln],[R("nums",a,[mid],M({lo:Math.min(lo,6),hi:Math.max(hi,0),mid}))]))}return f}});
prob({id:34,slug:"find-first-and-last-position-of-element-in-sorted-array",title:"Find First and Last Position of Element in Sorted Array",level:"Medium",topic:BSX,
x:["In a sorted list, find the first and last index of the target (−1, −1 if missing).","[5,7,7,8,8,10], target 8 → first at 3, last at 4.","Binary search for the left edge (bisect_left) and the right edge (bisect_right − 1)."],
i:"[5,7,7,8,8,10], target=8",o:"[3, 4]",
code:["class Solution:","    def searchRange(self, nums, target):","        lo = bisect_left(nums, target)","        if lo == len(nums) or nums[lo] != target: return [-1, -1]","        return [lo, bisect_right(nums, target) - 1]"],
time:"O(log n)",space:"O(1)",
frames:L=>{const a=[5,7,7,8,8,10];return [F("Sorted list","Looking for 8.",[L("def searchRange")],[R("nums",a)]),F("bisect_left","First position that is ≥ 8 → index 3.",[L("lo = bisect_left")],[R("nums",a,[3],M({first:3}))]),F("bisect_right","First position that is > 8 → index 5, so the last 8 is at 5 − 1 = 4.",[L("bisect_right")],[R("nums",a,[4],M({last:4}))]),F("Return","[3, 4]",[L("return [lo")],[R("answer",[3,4],[0,1])])]}});
prob({id:153,slug:"find-minimum-in-rotated-sorted-array",title:"Find Minimum in Rotated Sorted Array",level:"Medium",topic:BSX,
x:["A sorted list was rotated. Find the smallest value in O(log n).","[3,4,5,1,2] → the minimum is 1.","Compare the middle with the right end: if the middle is bigger, the minimum is to the right; otherwise it's at mid or to the left."],
i:"[3,4,5,1,2]",o:"1",
code:["class Solution:","    def findMin(self, nums):","        lo, hi = 0, len(nums) - 1","        while lo < hi:","            mid = (lo + hi) // 2","            if nums[mid] > nums[hi]: lo = mid + 1","            else: hi = mid","        return nums[lo]"],
time:"O(log n)",space:"O(1)",
frames:L=>{const a=[3,4,5,1,2];return [F("Start","Whole list.",[L("lo, hi")],[R("nums",a,[],M({lo:0,hi:4}))]),F("mid = 2","5 > 2 (right end) → the drop is to the right → lo = 3.",[L("if nums[mid] > nums[hi]")],[R("nums",a,[2,4],M({lo:0,mid:2,hi:4}))]),F("mid = 3","1 ≤ 2 → minimum is at mid or left → hi = 3.",[L("else: hi = mid")],[R("nums",a,[3,4],M({lo:3,mid:3,hi:4}))]),F("Return","lo = hi = 3 → value 1.",[L("return nums[lo]")],[R("nums",a,[3],M({min:3}))])]}});
prob({id:4,slug:"median-of-two-sorted-arrays",title:"Median of Two Sorted Arrays",level:"Hard",topic:BSX,
x:["Find the median of two sorted lists combined, in O(log(m+n)).","[1,3] and [2]: merged is 1,2,3 → median 2.","Binary search how many items to take from the smaller list so the left half and right half are balanced."],
i:"A=[1,3], B=[2]",o:"2.0",
code:["class Solution:","    def findMedianSortedArrays(self, A, B):","        if len(A) > len(B): A, B = B, A","        m, n, half = len(A), len(B), (len(A) + len(B) + 1) // 2","        lo, hi = 0, m","        while lo <= hi:","            i = (lo + hi) // 2; j = half - i","            a_left = A[i-1] if i else float('-inf'); a_right = A[i] if i < m else float('inf')","            b_left = B[j-1] if j else float('-inf'); b_right = B[j] if j < n else float('inf')","            if a_left <= b_right and b_left <= a_right:","                if (m + n) % 2: return max(a_left, b_left)","                return (max(a_left, b_left) + min(a_right, b_right)) / 2","            if a_left > b_right: hi = i - 1","            else: lo = i + 1"],
time:"O(log min(m,n))",space:"O(1)",
frames:L=>[F("Smaller list first","A = [2] (smaller), B = [1,3]. Left half should have 2 items in total.",[L("if len(A) > len(B)")],[R("A",[2]),R("B",[1,3])]),F("Try taking 0 from A","Left = [] + [1,3]; right = [2]. 3 > 2 → not balanced → take more from A.",[L("if a_left <= b_right and b_left <= a_right"),L("else: lo = i + 1")],[R("left",["B: 1","B: 3"]),R("right",["A: 2"])]),F("Take 1 from A","Left = [2] + [1]; right = [3]. Every left ≤ every right ✓.",[L("if a_left <= b_right")],[R("left",["A: 2","B: 1"],[0,1]),R("right",["B: 3"],[0])]),F("Median","Odd total → the biggest of the left side = 2.",[L("return max(a_left, b_left)")],[R("answer",["2.0"],[0])])]});
const HP="Heap";
prob({id:502,slug:"ipo",title:"IPO",level:"Hard",topic:HP,
x:["You start with capital w and can do at most k projects. Each project needs some minimum capital and gives a profit (added to your capital). Maximise final capital.","k=2, w=0, profits [1,2,3], capital [0,1,1] → do project 0 (+1 → w=1), then project 2 (+3 → w=4).","Each round, among projects you can afford, take the most profitable one (max-heap)."],
i:"k=2, w=0, profits=[1,2,3], capital=[0,1,1]",o:"4",
code:["class Solution:","    def findMaximizedCapital(self, k, w, profits, capital):","        projects = sorted(zip(capital, profits)); heap = []; i = 0","        for _ in range(k):","            while i < len(projects) and projects[i][0] <= w:","                heappush(heap, -projects[i][1]); i += 1","            if not heap: break","            w -= heappop(heap)","        return w"],
time:"O(n log n)",space:"O(n)",
frames:L=>[F("Sort by cost","(capital, profit): (0,1), (1,2), (1,3).",[L("projects = sorted")],[R("projects",["(0,1)","(1,2)","(1,3)"]),R("capital w",[0])]),F("Round 1","Only (0,1) is affordable → profit 1 → w = 1.",[L("while i < len(projects)"),L("w -= heappop(heap)")],[R("affordable profits",[1],[0]),R("capital w",[1])]),F("Round 2","Now (1,2) and (1,3) are affordable → take the best profit 3 → w = 4.",[L("w -= heappop(heap)")],[R("affordable profits",[3,2],[0]),R("capital w",[4])]),F("Return","k = 2 rounds done → 4.",[L("return w")],[R("answer",[4],[0])])]});
prob({id:373,slug:"find-k-pairs-with-smallest-sums",title:"Find K Pairs with Smallest Sums",level:"Medium",topic:HP,
x:["Two sorted lists. Pick one number from each to make a pair. List the k pairs with the smallest sums.","[1,7,11] and [2,4,6], k = 3 → (1,2), (1,4), (1,6).","A min-heap of candidate pairs; after taking (i, j), push the next pair (i, j+1)."],
i:"nums1=[1,7,11], nums2=[2,4,6], k=3",o:"[[1,2],[1,4],[1,6]]",
code:["class Solution:","    def kSmallestPairs(self, nums1, nums2, k):","        heap = [(nums1[i] + nums2[0], i, 0) for i in range(min(k, len(nums1)))]; heapify(heap)","        res = []","        while heap and len(res) < k:","            s, i, j = heappop(heap)","            res.append([nums1[i], nums2[j]])","            if j + 1 < len(nums2): heappush(heap, (nums1[i] + nums2[j+1], i, j + 1))","        return res"],
time:"O(k log k)",space:"O(k)",
frames:L=>[F("Start","Pair each nums1 value with the first of nums2: sums 3, 9, 13.",[L("heap = [(nums1[i]")],[R("heap (sum)",[3,9,13],[0]),R("result",[E])]),F("Pop (1,2)","Smallest sum 3. Push the next pair for 1: (1,4) = 5.",[L("res.append"),L("heappush")],[R("heap (sum)",[5,9,13],[0]),R("result",["[1,2]"],[0])]),F("Pop (1,4)","Sum 5. Push (1,6) = 7.",[L("res.append"),L("heappush")],[R("heap (sum)",[7,9,13],[0]),R("result",["[1,2]","[1,4]"],[1])]),F("Pop (1,6)","Sum 7 → we have k = 3 pairs.",[L("len(res) < k")],[R("result",["[1,2]","[1,4]","[1,6]"],[2])])]});
prob({id:295,slug:"find-median-from-data-stream",title:"Find Median from Data Stream",level:"Hard",topic:HP,
x:["Numbers arrive one at a time. After each, report the median so far.","Add 1, add 2 → median 1.5. Add 3 → median 2.","Keep the smaller half in a max-heap and the larger half in a min-heap; the median is at the tops."],
i:"addNum(1), addNum(2), findMedian(), addNum(3), findMedian()",o:"1.5, 2.0",
code:["class MedianFinder:","    def __init__(self): self.lo, self.hi = [], []","    def addNum(self, x):","        heappush(self.lo, -x); heappush(self.hi, -heappop(self.lo))","        if len(self.hi) > len(self.lo): heappush(self.lo, -heappop(self.hi))","    def findMedian(self):","        if len(self.lo) > len(self.hi): return -self.lo[0]","        return (-self.lo[0] + self.hi[0]) / 2"],
time:"O(log n) add, O(1) median",space:"O(n)",
frames:L=>[F("addNum(1)","Small half [1], big half [].",[L("heappush(self.lo")],[R("small half (max)",[1],[0]),R("big half (min)",[E])]),F("addNum(2)","Small [1], big [2].",[L("heappush(self.lo")],[R("small half",[1],[0]),R("big half",[2],[0])]),F("findMedian()","Equal sizes → (1 + 2) / 2 = 1.5",[L("return (-self.lo[0]")],[R("small half",[1],[0]),R("big half",[2],[0])],{median:1.5}),F("addNum(3)","Small [2,1], big [3].",[L("if len(self.hi) > len(self.lo)")],[R("small half",[2,1],[0]),R("big half",[3],[0])]),F("findMedian()","Small half is bigger → its top = 2.",[L("return -self.lo[0]")],[R("small half",[2,1],[0]),R("big half",[3])],{median:2})]});
const BX="Bit Manipulation";
prob({id:67,slug:"add-binary",title:"Add Binary",level:"Easy",topic:BX,
x:["Add two binary numbers given as strings.","'11' + '1' = 3 + 1 = 4 = '100'.","Add from the right with a carry, like adding on paper."],
i:'a="11", b="1"',o:'"100"',
code:["class Solution:","    def addBinary(self, a, b):","        i, j, carry, res = len(a) - 1, len(b) - 1, 0, []","        while i >= 0 or j >= 0 or carry:","            s = carry + (int(a[i]) if i >= 0 else 0) + (int(b[j]) if j >= 0 else 0)","            res.append(str(s % 2)); carry = s // 2; i -= 1; j -= 1","        return ''.join(reversed(res))"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Start","Add digit by digit from the right.",[L("i, j, carry")],[R("a",[1,1]),R("b",[0,1])]),F("1 + 1","= 2 → write 0, carry 1.",[L("s ="),L("res.append")],[R("a",[1,1],[1]),R("b",[0,1],[1]),R("result (reversed)",[0],[0])],{carry:1}),F("1 + 0 + carry 1","= 2 → write 0, carry 1.",[L("s =")],[R("a",[1,1],[0]),R("b",[0,1],[0]),R("result (reversed)",[0,0],[1])],{carry:1}),F("Carry only","1 → write 1.",[L("while i >= 0 or j >= 0 or carry")],[R("result (reversed)",[0,0,1],[2])],{carry:0}),F("Return","Reverse → '100'.",[L("return ''.join(reversed")],[R("answer","100",[0,1,2])])]});
prob({id:190,slug:"reverse-bits",title:"Reverse Bits",level:"Easy",topic:BX,
x:["Reverse the 32 bits of a number.","(Shown with 8 bits.) 00001011 reversed is 11010000.","Repeat: take the last bit of n and push it onto the end of the result."],
i:"00001011 (shown as 8 bits)",o:"11010000",
code:["class Solution:","    def reverseBits(self, n):","        res = 0","        for _ in range(32):","            res = (res << 1) | (n & 1)","            n >>= 1","        return res"],
time:"O(1)",space:"O(1)",
frames:L=>[F("Start","n = 00001011, result = 0.",[L("res = 0")],[R("n",[0,0,0,0,1,0,1,1]),R("result",[0,0,0,0,0,0,0,0])]),F("Step 1","Last bit of n is 1 → result gets 1.",[L("res = (res << 1) | (n & 1)")],[R("n",[0,0,0,0,0,1,0,1],[7]),R("result",[0,0,0,0,0,0,0,1],[7])]),F("Step 2","Last bit 1 → result 11.",[L("n >>= 1")],[R("n",[0,0,0,0,0,0,1,0],[7]),R("result",[0,0,0,0,0,0,1,1],[6,7])]),F("Step 3","Last bit 0 → result 110.",[L("res = (res << 1)")],[R("n",[0,0,0,0,0,0,0,1],[7]),R("result",[0,0,0,0,0,1,1,0],[5,6,7])]),F("Finish","After all bits: 11010000.",[L("return res")],[R("result",[1,1,0,1,0,0,0,0],range(0,7))])]});
prob({id:191,slug:"number-of-1-bits",title:"Number of 1 Bits",level:"Easy",topic:BX,
x:["Count the 1s in the binary form of a number.","11 is 1011 → three 1s.","n & (n−1) removes the lowest 1 bit. Count how many times you can do it."],
i:"11 (1011)",o:"3",
code:["class Solution:","    def hammingWeight(self, n):","        count = 0","        while n:","            n &= n - 1; count += 1","        return count"],
time:"O(#ones)",space:"O(1)",
frames:L=>[F("Start","n = 1011",[L("count = 0")],[R("n",[1,0,1,1])]),F("Clear a bit","1011 & 1010 = 1010 → count 1.",[L("n &= n - 1")],[R("n",[1,0,1,0],[3]),R("count",[1])]),F("Clear a bit","1010 & 1001 = 1000 → count 2.",[L("n &= n - 1")],[R("n",[1,0,0,0],[2]),R("count",[2])]),F("Clear a bit","1000 & 0111 = 0000 → count 3.",[L("n &= n - 1")],[R("n",[0,0,0,0],[0]),R("count",[3])]),F("Return","3",[L("return count")],[R("answer",[3],[0])])]});
prob({id:137,slug:"single-number-ii",title:"Single Number II",level:"Medium",topic:BX,
x:["Every number appears THREE times except one, which appears once. Find it.","[2,2,3,2] → 3.","Track which bits have been seen once ('ones') and twice ('twos'). A bit seen three times resets."],
i:"[2,2,3,2]",o:"3",
code:["class Solution:","    def singleNumber(self, nums):","        ones = twos = 0","        for x in nums:","            ones = (ones ^ x) & ~twos","            twos = (twos ^ x) & ~ones","        return ones"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","ones = twos = 0",[L("ones = twos = 0")],[R("nums",[2,2,3,2])],{ones:0,twos:0}),F("x = 2","First 2 → ones = 2.",[L("ones = (ones ^ x)")],[R("nums",[2,2,3,2],[0])],{ones:2,twos:0}),F("x = 2","Second 2 → moves to twos.",[L("twos = (twos ^ x)")],[R("nums",[2,2,3,2],[1])],{ones:0,twos:2}),F("x = 3","3 appears once → ones = 1 (bit 0); twos keeps bit for 2 (1 after combining).",[L("ones = (ones ^ x)")],[R("nums",[2,2,3,2],[2])],{ones:1,twos:0}),F("x = 2","Third 2 → combined with 3 → ones = 3.",[L("twos = (twos ^ x)")],[R("nums",[2,2,3,2],[3])],{ones:3,twos:0}),F("Return","3",[L("return ones")],[R("answer",[3],[0])])]});
prob({id:201,slug:"bitwise-and-of-numbers-range",title:"Bitwise AND of Numbers Range",level:"Medium",topic:BX,
x:["What is the bitwise AND of every number from left to right?","5 AND 6 AND 7 = 4.","The result is the common binary prefix of left and right. Shift both right until they match."],
i:"left=5 (101), right=7 (111)",o:"4",
code:["class Solution:","    def rangeBitwiseAnd(self, left, right):","        shift = 0","        while left != right:","            left >>= 1; right >>= 1; shift += 1","        return left << shift"],
time:"O(log n)",space:"O(1)",
frames:L=>[F("Start","left = 101, right = 111.",[L("shift = 0")],[R("left",[1,0,1]),R("right",[1,1,1])]),F("Shift 1","10 vs 11 — still different.",[L("left >>= 1")],[R("left",[1,0]),R("right",[1,1])],{shift:1}),F("Shift 2","1 vs 1 — equal! Common prefix = 1.",[L("while left != right")],[R("left",[1],[0]),R("right",[1],[0])],{shift:2}),F("Return","1 << 2 = 100 = 4.",[L("return left << shift")],[R("answer",[1,0,0],[0])])]});
const MT="Math";
prob({id:9,slug:"palindrome-number",title:"Palindrome Number",level:"Easy",topic:MT,
x:["Is the integer the same when its digits are reversed?","121 reads the same both ways → True. −121 isn't (the minus sign) → False.","Reverse only the second half of the digits and compare it with the first half."],
i:"121",o:"True",
code:["class Solution:","    def isPalindrome(self, x):","        if x < 0 or (x % 10 == 0 and x != 0): return False","        rev = 0","        while x > rev:","            rev = rev * 10 + x % 10; x //= 10","        return x == rev or x == rev // 10"],
time:"O(log n)",space:"O(1)",
frames:L=>[F("Start","x = 121, rev = 0.",[L("rev = 0")],[R("x",[121]),R("rev",[0])]),F("Move a digit","Take the last digit 1 → rev = 1, x = 12.",[L("rev = rev * 10")],[R("x",[12]),R("rev",[1])]),F("Move a digit","Take 2 → rev = 12, x = 1. Now x ≤ rev → stop.",[L("while x > rev")],[R("x",[1]),R("rev",[12])]),F("Compare","x = 1 equals rev // 10 = 1 (middle digit ignored) → True.",[L("return x == rev")],[R("answer",["True"],[0])])]});
prob({id:66,slug:"plus-one",title:"Plus One",level:"Easy",topic:MT,
x:["A number is stored as a list of digits. Add one to it.","[1,2,9] means 129 → 130 → [1,3,0].","From the right: a 9 becomes 0 and carries; the first non-9 digit gets +1."],
i:"[1,2,9]",o:"[1,3,0]",
code:["class Solution:","    def plusOne(self, digits):","        for i in range(len(digits) - 1, -1, -1):","            if digits[i] < 9:","                digits[i] += 1; return digits","            digits[i] = 0","        return [1] + digits"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","Look at the last digit.",[L("for i in range")],[R("digits",[1,2,9],[2])]),F("9 → 0","9 can't grow → set to 0 and carry.",[L("digits[i] = 0")],[R("digits",[1,2,0],[2])]),F("2 → 3","2 < 9 → add 1 and stop.",[L("digits[i] += 1")],[R("digits",[1,3,0],[1])]),F("Return","[1, 3, 0]",[L("return digits")],[R("answer",[1,3,0],[0,1,2])])]});
prob({id:172,slug:"factorial-trailing-zeroes",title:"Factorial Trailing Zeroes",level:"Medium",topic:MT,
x:["How many zeros are at the end of n! (n factorial)?","25! ends in 6 zeros.","Each zero comes from a factor 10 = 2 × 5, and 5s are rarer than 2s. Count the 5s: n/5 + n/25 + …"],
i:"25",o:"6",
code:["class Solution:","    def trailingZeroes(self, n):","        count = 0","        while n:","            n //= 5; count += n","        return count"],
time:"O(log n)",space:"O(1)",
frames:L=>[F("Start","Count factors of 5 in 1…25.",[L("count = 0")],[R("n",[25]),R("count",[0])]),F("Multiples of 5","25 ÷ 5 = 5 numbers (5,10,15,20,25) → count 5.",[L("n //= 5; count += n")],[R("n",[5]),R("count",[5])]),F("Multiples of 25","5 ÷ 5 = 1 more (25 has a second 5) → count 6.",[L("n //= 5; count += n")],[R("n",[1]),R("count",[6])]),F("Return","6",[L("return count")],[R("answer",[6],[0])])]});
prob({id:69,slug:"sqrt-x",title:"Sqrt(x)",level:"Easy",topic:MT,
x:["Compute the square root of x, rounded down to a whole number (no built-in sqrt).","sqrt(8) = 2.82… → 2.","Binary search for the biggest number whose square is ≤ x."],
i:"8",o:"2",
code:["class Solution:","    def mySqrt(self, x):","        lo, hi = 0, x","        while lo <= hi:","            mid = (lo + hi) // 2","            if mid * mid <= x: lo = mid + 1","            else: hi = mid - 1","        return hi"],
time:"O(log x)",space:"O(1)",
frames:L=>[F("Range","Answer is between 0 and 8.",[L("lo, hi")],[R("candidates",range(0,8),[],M({lo:0,hi:8}))]),F("mid = 4","16 > 8 → too big → hi = 3.",[L("else: hi = mid - 1")],[R("candidates",range(0,8),[4])],{lo:0,hi:3}),F("mid = 1","1 ≤ 8 → fine → lo = 2.",[L("if mid * mid <= x")],[R("candidates",range(0,8),[1])],{lo:2,hi:3}),F("mid = 2","4 ≤ 8 → lo = 3.",[L("if mid * mid <= x")],[R("candidates",range(0,8),[2])],{lo:3,hi:3}),F("mid = 3","9 > 8 → hi = 2. Now lo > hi.",[L("else: hi = mid - 1")],[R("candidates",range(0,8),[3])],{lo:3,hi:2}),F("Return","hi = 2",[L("return hi")],[R("answer",[2],[0])])]});
prob({id:50,slug:"powx-n",title:"Pow(x, n)",level:"Medium",topic:MT,
x:["Compute x raised to the power n quickly.","2^10 = 1024.","Repeated squaring: look at the bits of n; multiply the result by x when the bit is 1, and square x each step."],
i:"x=2, n=10",o:"1024",
code:["class Solution:","    def myPow(self, x, n):","        if n < 0: x, n = 1 / x, -n","        res = 1","        while n:","            if n & 1: res *= x","            x *= x; n >>= 1","        return res"],
time:"O(log n)",space:"O(1)",
frames:L=>[F("Start","n = 10 = 1010 in binary.",[L("res = 1")],[R("n bits",[1,0,1,0]),R("x",[2]),R("res",[1])]),F("Bit 0 = 0","Skip; x = 4.",[L("x *= x; n >>= 1")],[R("n bits",[1,0,1],[2]),R("x",[4]),R("res",[1])]),F("Bit = 1","res = 1 × 4 = 4; x = 16.",[L("if n & 1: res *= x")],[R("n bits",[1,0],[1]),R("x",[16]),R("res",[4])]),F("Bit = 0","Skip; x = 256.",[L("x *= x")],[R("n bits",[1],[0]),R("x",[256]),R("res",[4])]),F("Bit = 1","res = 4 × 256 = 1024.",[L("res *= x")],[R("n bits",[E]),R("res",[1024],[0])])]});
prob({id:149,slug:"max-points-on-a-line",title:"Max Points on a Line",level:"Hard",topic:MT,
x:["Given points on a plane, what's the most points that lie on one straight line?","Points (1,1), (2,2), (3,3), (1,4): the first three are on a line → 3.","From each point, count other points by their slope; the biggest group + the point itself is a line."],
i:"[[1,1],[2,2],[3,3],[1,4]]",o:"3",
code:["class Solution:","    def maxPoints(self, points):","        best = 1","        for i, (x1, y1) in enumerate(points):","            slopes = Counter()","            for x2, y2 in points[i+1:]:","                g = gcd(y2 - y1, x2 - x1)","                dy, dx = (y2 - y1) // g, (x2 - x1) // g","                if dx < 0 or (dx == 0 and dy < 0): dy, dx = -dy, -dx","                slopes[(dy, dx)] += 1","            if slopes: best = max(best, max(slopes.values()) + 1)","        return best"],
time:"O(n²)",space:"O(n)",
frames:L=>[F("Points","From (1,1), measure the slope to each other point.",[L("for i, (x1, y1)")],[R("points",["(1,1)","(2,2)","(3,3)","(1,4)"],[0])]),F("Slopes","(2,2) → (1,1); (3,3) → (1,1); (1,4) → (1,0). Reduce with gcd.",[L("g = gcd"),L("slopes[")],[R("slope → count",["(1,1): 2","(1,0): 1"],[0])]),F("Best group","2 points share a slope + the start point = 3.",[L("best = max")],[R("on one line",["(1,1)","(2,2)","(3,3)"],[0,1,2])],{best:3}),F("Return","3",[L("return best")],[R("answer",[3],[0])])]});
const D1="1D DP";
prob({id:70,slug:"climbing-stairs",title:"Climbing Stairs",level:"Easy",topic:D1,
x:["You climb a staircase taking 1 or 2 steps at a time. How many different ways to reach step n?","n = 5 → 8 ways.","Ways(n) = Ways(n−1) + Ways(n−2) — it's the Fibonacci sequence."],
i:"n = 5",o:"8",
code:["class Solution:","    def climbStairs(self, n):","        a, b = 1, 1","        for _ in range(n - 1): a, b = b, a + b","        return b"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","1 way to be on step 0, and 1 way to reach step 1.",[L("a, b = 1, 1")],[R("ways",[1,1],[0,1])]),F("Step 2","1 + 1 = 2",[L("a, b = b, a + b")],[R("ways",[1,1,2],[2])]),F("Step 3","1 + 2 = 3",[L("a, b = b, a + b")],[R("ways",[1,1,2,3],[3])]),F("Step 4","2 + 3 = 5",[L("a, b = b, a + b")],[R("ways",[1,1,2,3,5],[4])]),F("Step 5","3 + 5 = 8",[L("a, b = b, a + b")],[R("ways",[1,1,2,3,5,8],[5])]),F("Return","8",[L("return b")],[R("answer",[8],[0])])]});
prob({id:139,slug:"word-break",title:"Word Break",level:"Medium",topic:D1,
x:["Can the text be split into words from the dictionary?","'leetcode' = 'leet' + 'code' → True.","dp[i] = True if some earlier dp[j] is True and s[j:i] is a word."],
i:'s="leetcode", wordDict=["leet","code"]',o:"True",
code:["class Solution:","    def wordBreak(self, s, wordDict):","        words = set(wordDict); dp = [True] + [False] * len(s)","        for i in range(1, len(s) + 1):","            dp[i] = any(dp[j] and s[j:i] in words for j in range(i))","        return dp[-1]"],
time:"O(n²)",space:"O(n)",
frames:L=>{const s="leetcode",dp=[1,0,0,0,1,0,0,0,1];return [F("Start","dp[0] = True (empty prefix is fine). dp[i] means s[:i] can be split.",[L("dp = [True]")],[R("s",s),R("dp",[1,0,0,0,0,0,0,0,0],[0])]),F("i = 4","s[0:4] = 'leet' is a word and dp[0] is True → dp[4] = True.",[L("dp[i] = any")],[R("s",s,[0,1,2,3]),R("dp",[1,0,0,0,1,0,0,0,0],[4])]),F("i = 8","s[4:8] = 'code' is a word and dp[4] is True → dp[8] = True.",[L("dp[i] = any")],[R("s",s,[4,5,6,7]),R("dp",dp,[8])]),F("Return","dp[8] is True.",[L("return dp[-1]")],[R("answer",["True"],[0])])]}});
prob({id:322,slug:"coin-change",title:"Coin Change",level:"Medium",topic:D1,
x:["With unlimited coins of given values, what's the fewest coins that make the amount? (−1 if impossible)","coins [1,2,5], amount 11 → 5 + 5 + 1 = 3 coins.","dp[a] = 1 + the best of dp[a − coin] over all coins."],
i:"coins=[1,2,5], amount=11",o:"3",
code:["class Solution:","    def coinChange(self, coins, amount):","        dp = [0] + [float('inf')] * amount","        for a in range(1, amount + 1):","            for c in coins:","                if c <= a: dp[a] = min(dp[a], dp[a - c] + 1)","        return dp[amount] if dp[amount] != float('inf') else -1"],
time:"O(amount·coins)",space:"O(amount)",
frames:L=>{const co=[1,2,5],dp=[0];for(let a=1;a<=11;a++){let b=Infinity;for(const c of co)if(c<=a)b=Math.min(b,dp[a-c]+1);dp.push(b)}return [F("Start","dp[a] = fewest coins to make a. dp[0] = 0.",[L("dp = [0]")],[R("amount",range(0,11)),R("coins needed",[0,"∞","∞","∞","∞","∞","∞","∞","∞","∞","∞","∞"].slice(0,12),[0])]),F("Small amounts","dp[1]=1, dp[2]=1, dp[3]=2, dp[4]=2, dp[5]=1.",[L("dp[a] = min")],[R("amount",range(0,5)),R("coins needed",dp.slice(0,6),[5])]),F("Up to 10","dp[6]=2 (5+1), dp[7]=2 (5+2), dp[8]=3, dp[9]=3, dp[10]=2 (5+5).",[L("dp[a] = min")],[R("amount",range(0,11)),R("coins needed",dp.slice(0,11),[10])]),F("dp[11]","dp[10] + 1 = 3 (5 + 5 + 1).",[L("dp[a] = min")],[R("amount",range(0,12)),R("coins needed",dp,[11])]),F("Return","3",[L("return dp[amount]")],[R("answer",[3],[0])])]}});
prob({id:300,slug:"longest-increasing-subsequence",title:"Longest Increasing Subsequence",level:"Medium",topic:D1,
x:["Find the length of the longest strictly increasing subsequence (numbers in order, not necessarily next to each other).","[10,9,2,5,3,7,101,18] → 2,3,7,101 → length 4.","Keep a list 'tails' where tails[k] is the smallest possible end of an increasing subsequence of length k+1. Replace using binary search."],
i:"[10,9,2,5,3,7,101,18]",o:"4",
code:["class Solution:","    def lengthOfLIS(self, nums):","        tails = []","        for x in nums:","            i = bisect_left(tails, x)","            if i == len(tails): tails.append(x)","            else: tails[i] = x","        return len(tails)"],
time:"O(n log n)",space:"O(n)",
frames:L=>{const a=[10,9,2,5,3,7,101,18],t=[],f=[F("Start","tails is empty.",[L("tails = []")],[R("nums",a),R("tails",[E])])];
 a.forEach((x,k)=>{let i=0;while(i<t.length&&t[i]<x)i++;let d;if(i===t.length){t.push(x);d="Bigger than all → extend (length "+t.length+")."}else{t[i]=x;d="Replace "+(t[i]===x?"slot "+i:"")+" with a smaller end."}f.push(F("x = "+x,d,[L("i = bisect_left"),L(i===t.length-1&&t[i]===x&&d.startsWith("Bigger")?"tails.append(x)":"else: tails[i] = x")],[R("nums",a,[k]),R("tails",t,[i])]))});
 f.push(F("Return","len(tails) = "+t.length,[L("return len(tails)")],[R("answer",[t.length],[0])]));return f}});
const D2="Multidimensional DP";
prob({id:120,slug:"triangle",title:"Triangle",level:"Medium",topic:D2,
x:["Walk from the top of a number triangle to the bottom, moving to an adjacent number each row. What's the smallest path sum?","[[2],[3,4],[6,5,7],[4,1,8,3]] → 2 + 3 + 5 + 1 = 11.","Work from the bottom up: each cell = its value + the smaller of the two cells below it."],
i:"[[2],[3,4],[6,5,7],[4,1,8,3]]",o:"11",
code:["class Solution:","    def minimumTotal(self, triangle):","        dp = triangle[-1][:]","        for row in range(len(triangle) - 2, -1, -1):","            for i in range(len(triangle[row])):","                dp[i] = triangle[row][i] + min(dp[i], dp[i + 1])","        return dp[0]"],
time:"O(n²)",space:"O(n)",
frames:L=>[F("Bottom row","Start with the last row: [4,1,8,3].",[L("dp = triangle[-1]")],[R("dp",[4,1,8,3])]),F("Row [6,5,7]","6+min(4,1)=7, 5+min(1,8)=6, 7+min(8,3)=10",[L("dp[i] = triangle[row][i] + min")],[R("dp",[7,6,10],[0,1,2])]),F("Row [3,4]","3+min(7,6)=9, 4+min(6,10)=10",[L("dp[i] = triangle[row][i] + min")],[R("dp",[9,10],[0,1])]),F("Top","2 + min(9,10) = 11",[L("dp[i] = triangle[row][i] + min")],[R("dp",[11],[0])]),F("Return","11",[L("return dp[0]")],[R("answer",[11],[0])])]});
prob({id:64,slug:"minimum-path-sum",title:"Minimum Path Sum",level:"Medium",topic:D2,
x:["Move from the top-left to the bottom-right of a grid, only right or down. What's the smallest sum of the numbers you pass?","[[1,3,1],[1,5,1],[4,2,1]] → 1 → 3 → 1 → 1 → 1 = 7.","Each cell = its value + the smaller of (cell above, cell to the left)."],
i:"[[1,3,1],[1,5,1],[4,2,1]]",o:"7",
code:["class Solution:","    def minPathSum(self, grid):","        m, n = len(grid), len(grid[0])","        for i in range(m):","            for j in range(n):","                if i == 0 and j == 0: continue","                up = grid[i-1][j] if i else float('inf'); left = grid[i][j-1] if j else float('inf')","                grid[i][j] += min(up, left)","        return grid[-1][-1]"],
time:"O(m·n)",space:"O(1)",
frames:L=>[F("Grid","Cheapest way to reach each cell.",[L("def minPathSum")],[GR("grid",[[1,3,1],[1,5,1],[4,2,1]])]),F("Row 0","Only from the left: 1, 4, 5.",[L("grid[i][j] += min(up, left)")],[GR("grid",[[1,4,5],[1,5,1],[4,2,1]],["0,0","0,1","0,2"])]),F("Row 1","1+1 = 2; 5+min(4,2) = 7; 1+min(5,7) = 6.",[L("grid[i][j] += min(up, left)")],[GR("grid",[[1,4,5],[2,7,6],[4,2,1]],["1,0","1,1","1,2"])]),F("Row 2","4+2 = 6; 2+min(7,6) = 8; 1+min(6,8) = 7.",[L("grid[i][j] += min(up, left)")],[GR("grid",[[1,4,5],[2,7,6],[6,8,7]],["2,0","2,1","2,2"])]),F("Return","Bottom-right = 7.",[L("return grid[-1][-1]")],[GR("grid",[[1,4,5],[2,7,6],[6,8,7]],["2,2"])])]});
prob({id:63,slug:"unique-paths-ii",title:"Unique Paths II",level:"Medium",topic:D2,
x:["Like Unique Paths, but some cells are obstacles (1) you can't step on. How many paths from top-left to bottom-right?","With one obstacle in the middle of a 3×3 grid there are 2 paths.","Paths into a cell = paths from above + paths from the left; obstacle cells have 0."],
i:"[[0,0,0],[0,1,0],[0,0,0]]",o:"2",
code:["class Solution:","    def uniquePathsWithObstacles(self, g):","        m, n = len(g), len(g[0]); dp = [0] * n; dp[0] = 1","        for i in range(m):","            for j in range(n):","                if g[i][j] == 1: dp[j] = 0","                elif j: dp[j] += dp[j - 1]","        return dp[-1]"],
time:"O(m·n)",space:"O(n)",
frames:L=>[F("Grid","1 marks an obstacle in the center.",[L("def unique")],[GR("grid",[[0,0,0],[0,1,0],[0,0,0]],["1,1"])]),F("Row 0","One way to reach each cell: 1, 1, 1.",[L("dp[j] += dp[j - 1]")],[GR("paths",[[1,1,1],[E,E,E],[E,E,E]],["0,0","0,1","0,2"])]),F("Row 1","1, then the obstacle → 0, then 0 + 1 = 1.",[L("if g[i][j] == 1: dp[j] = 0")],[GR("paths",[[1,1,1],[1,0,1],[E,E,E]],["1,1"])]),F("Row 2","1; 0 + 1 = 1; 1 + 1 = 2.",[L("dp[j] += dp[j - 1]")],[GR("paths",[[1,1,1],[1,0,1],[1,1,2]],["2,2"])]),F("Return","2 paths.",[L("return dp[-1]")],[R("answer",[2],[0])])]});
prob({id:5,slug:"longest-palindromic-substring",title:"Longest Palindromic Substring",level:"Medium",topic:D2,
x:["Find the longest piece of the text that reads the same backwards.","'babad' → 'bab' (or 'aba') → length 3.","Treat every position (and every gap) as a center and expand outwards while the letters match."],
i:'"babad"',o:'"bab"',
code:["class Solution:","    def longestPalindrome(self, s):","        best = ''","        for c in range(len(s)):","            for l, r in ((c, c), (c, c + 1)):","                while l >= 0 and r < len(s) and s[l] == s[r]: l -= 1; r += 1","                if r - l - 1 > len(best): best = s[l+1:r]","        return best"],
time:"O(n²)",space:"O(1)",
frames:L=>[F("Start","Try each letter as a center.",[L("for c in range")],[R("s","babad")]),F("Center 'b' (0)","Can't expand left → 'b'.",[L("while l >= 0")],[R("s","babad",[0])],{best:"b"}),F("Center 'a' (1)","Expand: b = b → 'bab'.",[L("s[l] == s[r]"),L("best = s[l+1:r]")],[R("s","babad",[0,1,2])],{best:"bab"}),F("Other centers","Centers at 2, 3, 4 give 'aba' (length 3, not longer), 'd', etc.",[L("if r - l - 1 > len(best)")],[R("s","babad",[1,2,3])],{best:"bab"}),F("Return","'bab'",[L("return best")],[R("answer","bab",[0,1,2])])]});
prob({id:97,slug:"interleaving-string",title:"Interleaving String",level:"Medium",topic:D2,
x:["Can s3 be made by weaving letters of s1 and s2 together, keeping each string's own order?","'ab' and 'cd' can make 'acbd' (a from s1, c from s2, b from s1, d from s2) → True.","dp[i][j] = can the first i letters of s1 and first j of s2 make the first i+j letters of s3?"],
i:'s1="ab", s2="cd", s3="acbd"',o:"True",
code:["class Solution:","    def isInterleave(self, s1, s2, s3):","        if len(s1) + len(s2) != len(s3): return False","        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]; dp[0][0] = True","        for i in range(len(s1) + 1):","            for j in range(len(s2) + 1):","                if i and dp[i-1][j] and s1[i-1] == s3[i+j-1]: dp[i][j] = True","                if j and dp[i][j-1] and s2[j-1] == s3[i+j-1]: dp[i][j] = True","        return dp[-1][-1]"],
time:"O(n·m)",space:"O(n·m)",
frames:L=>[F("Table","Rows = letters of s1 used, columns = letters of s2 used. 1 = possible.",[L("dp = [[False]")],[GR("dp",[[1,0,0],[0,0,0],[0,0,0]],["0,0"])]),F("Use s1's 'a'","Take 'a' from s1 → matches s3[0] = 'a'.",[L("if i and dp[i-1][j]")],[GR("dp",[[1,0,0],[1,0,0],[0,0,0]],["1,0"])]),F("Then s2's 'c'","Take 'c' from s2 → matches s3[1] = 'c'.",[L("if j and dp[i][j-1]")],[GR("dp",[[1,0,0],[1,1,0],[0,0,0]],["1,1"])]),F("Then s1's 'b'","Take 'b' from s1 → matches s3[2] = 'b'.",[L("if i and dp[i-1][j]")],[GR("dp",[[1,0,0],[1,1,0],[0,1,0]],["2,1"])]),F("Then s2's 'd'","Take 'd' from s2 → matches s3[3] = 'd'. Bottom-right is 1 → True.",[L("return dp[-1][-1]")],[GR("dp",[[1,0,0],[1,1,0],[0,1,1]],["2,2"])])]});
prob({id:123,slug:"best-time-to-buy-and-sell-stock-iii",title:"Best Time to Buy and Sell Stock III",level:"Hard",topic:D2,
x:["You may complete at most TWO buy-sell trades. What's the maximum profit?","[3,3,5,0,0,3,1,4]: buy 0 sell 3 (+3), buy 1 sell 4 (+3) → 6.","Track four numbers: best after buy 1, after sell 1, after buy 2, after sell 2."],
i:"[3,3,5,0,0,3,1,4]",o:"6",
code:["class Solution:","    def maxProfit(self, prices):","        b1 = b2 = float('-inf'); s1 = s2 = 0","        for p in prices:","            b1 = max(b1, -p); s1 = max(s1, b1 + p)","            b2 = max(b2, s1 - p); s2 = max(s2, b2 + p)","        return s2"],
time:"O(n)",space:"O(1)",
frames:L=>{const p=[3,3,5,0,0,3,1,4];let b1=-Infinity,b2=-Infinity,s1=0,s2=0;const f=[F("Start","Four running values.",[L("b1 = b2")],[R("prices",p)])];
 p.forEach((x,i)=>{b1=Math.max(b1,-x);s1=Math.max(s1,b1+x);b2=Math.max(b2,s1-x);s2=Math.max(s2,b2+x);f.push(F("Day "+i+" (price "+x+")","After trade 1: "+s1+" · after trade 2: "+s2,[L("b1 = max"),L("b2 = max")],[R("prices",p,[i])],{profit1:s1,profit2:s2}))});
 f.push(F("Return","Best with two trades = "+s2+".",[L("return s2")],[R("answer",[s2],[0])]));return f}});
prob({id:188,slug:"best-time-to-buy-and-sell-stock-iv",title:"Best Time to Buy and Sell Stock IV",level:"Hard",topic:D2,
x:["You may complete at most k buy-sell trades. What's the maximum profit?","k = 2, prices [3,2,6,5,0,3]: buy 2 sell 6 (+4), buy 0 sell 3 (+3) → 7.","Like the two-trade version, but keep buy[t] and sell[t] for every trade count t up to k."],
i:"k=2, prices=[3,2,6,5,0,3]",o:"7",
code:["class Solution:","    def maxProfit(self, k, prices):","        buy = [float('-inf')] * (k + 1); sell = [0] * (k + 1)","        for p in prices:","            for t in range(1, k + 1):","                buy[t] = max(buy[t], sell[t-1] - p)","                sell[t] = max(sell[t], buy[t] + p)","        return sell[k]"],
time:"O(n·k)",space:"O(k)",
frames:L=>{const p=[3,2,6,5,0,3],k=2,buy=[-Infinity,-Infinity,-Infinity],sell=[0,0,0];const f=[F("Start","sell[t] = best profit with t trades.",[L("buy = [float")],[R("prices",p),R("sell",[0,0,0])])];
 p.forEach((x,i)=>{for(let t=1;t<=k;t++){buy[t]=Math.max(buy[t],sell[t-1]-x);sell[t]=Math.max(sell[t],buy[t]+x)}f.push(F("Day "+i+" (price "+x+")","sell[1] = "+sell[1]+", sell[2] = "+sell[2],[L("buy[t] = max"),L("sell[t] = max")],[R("prices",p,[i]),R("sell",sell.slice(),[1,2])]))});
 f.push(F("Return","sell[2] = "+sell[2]+".",[L("return sell[k]")],[R("answer",[sell[2]],[0])]));return f}});
prob({id:221,slug:"maximal-square",title:"Maximal Square",level:"Medium",topic:D2,
x:["In a grid of 0s and 1s, find the largest square made only of 1s and return its AREA.","The largest all-1 square in the sample is 2×2 → area 4.","dp[i][j] = side of the biggest square ending at that cell = 1 + min(up, left, up-left)."],
i:"4×5 grid of 1s and 0s",o:"4",
code:["class Solution:","    def maximalSquare(self, m):","        rows, cols = len(m), len(m[0]); dp = [[0] * (cols + 1) for _ in range(rows + 1)]; best = 0","        for i in range(1, rows + 1):","            for j in range(1, cols + 1):","                if m[i-1][j-1] == '1':","                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]); best = max(best, dp[i][j])","        return best * best"],
time:"O(m·n)",space:"O(m·n)",
frames:L=>{const m=[[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]];const dp=Array.from({length:5},()=>Array(6).fill(0));let best=0;const f=[F("Grid","1 = filled.",[L("def maximalSquare")],[GR("grid",m)])];
 for(let i=1;i<=4;i++){const hot=[];for(let j=1;j<=5;j++)if(m[i-1][j-1]){dp[i][j]=1+Math.min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]);best=Math.max(best,dp[i][j]);if(dp[i][j]>=2)hot.push(i+","+j)}f.push(F("Row "+i,"Square side ending at each cell.",[L("dp[i][j] = 1 + min")],[GR("dp",dp.map(r=>r.slice()),hot)],{best}))}
 f.push(F("Return","Biggest side = 2 → area 4.",[L("return best * best")],[R("answer",[best*best],[0])]));return f}});
