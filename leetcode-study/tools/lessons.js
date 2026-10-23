/* LeetCode Study Gallery — lessons based on our chat */
(function(){
const lines = a => a.join("\n");
const row = (name, cells, hot=[]) => ({name, cells: Array.isArray(cells)?cells:Array.from(cells), hot});
const frame = (title, description, rows, codeLines=[], facts={}) => ({title, description, rows, codeLines, facts});
const lesson=(id,title,topic,notes,variants,recommendation="")=>({id,title,topic,notes,variants,recommendation});
const variant=(name,code,time,space,frames,explain)=>({name,code:lines(code),time,space,frames,explain});
window.STUDY_LESSONS=[
lesson("1071","Greatest Common Divisor of Strings","Strings",[
"n1 = len(str1)//k and n2 = len(str2)//k are the repetition counts.",
"String multiplication repeats characters: 'ABC' * 3 becomes 'ABCABCABC'.",
"Strings concatenation + GCD avoids checking candidate lengths one by one.",
"String slicing and repetition cost time and allocate memory proportional to their length."
],[
variant("Candidate scan",[
"class Solution:",
"    def gcdOfStrings(self, str1, str2):",
"        len1, len2 = len(str1), len(str2)",
"        def valid(k):",
"            if len1 % k or len2 % k: return False",
"            n1, n2 = len1 // k, len2 // k",
"            base = str1[:k]",
"            return str1 == n1*base and str2 == n2*base",
"        for i in range(min(len1, len2), 0, -1):",
"            if valid(i): return str1[:i]",
"        return ''"
],"O(min(n,m) × (n+m)) upper bound","O(n+m)",[
frame("Input","Find the longest string that can repeat to form both inputs.",[row("str1","ABCABCABC"),row("str2","ABCABC")],[2,3]),
frame("Check k = 6, 5, 4","At least one length is not divisible by each candidate, so skip.",[row("candidates",[6,5,4],[0,1,2])],[5,9]),
frame("Check k = 3","Base ABC; repeat it 3 times for str1 and 2 times for str2.",[row("base","ABC",[0,1,2]),row("3 copies","ABCABCABC"),row("2 copies","ABCABC")],[6,7,8]),
frame("Return","Both reconstructions match, so the answer is ABC.",[row("result","ABC",[0,1,2])],[10])
],"Straightforward verification but revisits many candidates."),
variant("Concatenate + GCD",[
"from math import gcd",
"class Solution:",
"    def gcdOfStrings(self, str1, str2):",
"        if str1 + str2 != str2 + str1:",
"            return ''",
"        max_length = gcd(len(str1), len(str2))",
"        return str1[:max_length]"
],"O(n+m)","O(n+m)",[
frame("Input","Both strings are repetitions of ABC.",[row("str1","ABCABCABC"),row("str2","ABCABC")],[3]),
frame("Commutativity check","Concatenating in either order gives exactly the same string.",[row("str1+str2","ABCABCABCABCABC"),row("str2+str1","ABCABCABCABCABC")],[4,5]),
frame("Calculate length","gcd(9,6) = 3, so the longest valid base has length 3.",[row("lengths",[9,6]),row("GCD",[3],[0])],[6]),
frame("Return prefix","Take str1[:3].",[row("result","ABC",[0,1,2])],[7])
],"Compact mathematical solution with a single linear compatibility check.")
],"Concatenate + GCD is better: O(n+m) time instead of an expensive candidate scan."),
lesson("1431","Kids With the Greatest Number of Candies","Arrays",[
"max(candies) requires O(n) time even though it is one Python call.",
"To implement max yourself, start with candies[0] rather than zero (negative numbers!).",
"Two consecutive O(n) scans still total O(n), not O(n²).",
"Comparison inside append produces a Boolean (True/False)."
],[
variant("Built-in max()",[
"class Solution:",
"    def kidsWithCandies(self, candies, extraCandies):",
"        maxCandies = max(candies)",
"        result = []",
"        for i in range(len(candies)):",
"            result.append(candies[i] + extraCandies >= maxCandies)",
"        return result"
],"O(n)","O(n) output; O(1) auxiliary",[
frame("Start","Each child hypothetically receives all 3 extra candies.",[row("candies",[2,3,5,1,3])],[2]),
frame("Find maximum","max([2,3,5,1,3]) equals 5.",[row("candies",[2,3,5,1,3],[2])],[3]),
frame("Check totals","Add 3 to each value independently.",[row("totals",[5,6,8,4,6],[0,1,2,4])],[5,6]),
frame("Return Booleans","True means the child can reach at least the original maximum.",[row("answer",["T","T","T","F","T"],[0,1,2,4])],[6,7])
],"Most readable for normal Python."),
variant("Manual max",[
"class Solution:",
"    def kidsWithCandies(self, candies, extraCandies):",
"        biggest = candies[0]",
"        for value in candies:",
"            if value > biggest: biggest = value",
"        result = []",
"        for value in candies:",
"            result.append(value + extraCandies >= biggest)",
"        return result"
],"O(n)","O(n) output; O(1) auxiliary",[
frame("Start with first item","biggest = candies[0] = 2.",[row("candies",[2,3,5,1,3],[0])],[3]),
frame("Scan and update","Observe 3, then 5; running maximum becomes 5.",[row("running max",[2,3,5,5,5],[2])],[4,5]),
frame("Check each child","Compare candy + 3 >= 5.",[row("totals",[5,6,8,4,6],[0,1,2,4])],[7,8]),
frame("Return","Same Boolean answer as max().",[row("answer",["T","T","T","F","T"])],[9])
],"Shows the logic inside Python's max() helper.")
],"Built-in max() is shorter and equally efficient. Manual max() is useful for learning."),
lesson("605","Can Place Flowers","Greedy",[
"Greedy means take the earliest locally valid planting decision, without backtracking.",
"OR short circuits: i==0 prevents accidentally looking up flowerbed[-1].",
"AND ensures both neighbors are empty (or outside the boundaries).",
"Early return improves practical speed but worst-case time remains O(m).",
"Check n == 0 first to avoid unnecessary mutation of flowerbed."
],[
variant("Greedy + early return",[
"class Solution:",
"    def canPlaceFlowers(self, flowerbed, n):",
"        if n == 0: return True",
"        count = 0",
"        for i in range(len(flowerbed)):",
"            if flowerbed[i] == 0:",
"                left = i == 0 or flowerbed[i-1] == 0",
"                right = i == len(flowerbed)-1 or flowerbed[i+1] == 0",
"                if left and right:",
"                    flowerbed[i] = 1",
"                    count += 1",
"                    if count >= n: return True",
"        return False"
],"O(m)","O(1) auxiliary",[
frame("Input","Need to plant one flower without adjacent flowers.",[row("plots",[1,0,0,0,1])],[2,3]),
frame("i=0: skip","The first plot is already occupied.",[row("plots",[1,0,0,0,1],[0])],[5,6]),
frame("i=1: blocked","Index 1 is empty but left neighbor index 0 contains 1.",[row("plots",[1,0,0,0,1],[0,1])],[7,8]),
frame("i=2: can plant","Index 2 and both neighbors are 0.",[row("plots",[1,0,0,0,1],[1,2,3])],[7,8,9]),
frame("Plant and return","Change index 2 to 1. count = 1, so return True.",[row("plots",[1,0,1,0,1],[2])],[10,11,12])
],"Choose the earliest legal position and stop once enough flowers are planted.")
]),
lesson("345","Reverse Vowels of a String","Two Pointers",[
"Two pointers scan inward from the left and right ends.",
"Skip consonants; swap only vowel characters.",
"Python strings are immutable, so list(s) allows indexed swapping.",
"Time O(n); the Python list and resulting string require O(n) memory."
],[
variant("Two pointers",[
"class Solution:",
"    def reverseVowels(self, s):",
"        vowels = set('aeiouAEIOU')",
"        a = list(s)",
"        left, right = 0, len(a)-1",
"        while left < right:",
"            while left < right and a[left] not in vowels: left += 1",
"            while left < right and a[right] not in vowels: right -= 1",
"            a[left], a[right] = a[right], a[left]",
"            left, right = left+1, right-1",
"        return ''.join(a)"
],"O(n)","O(n)",[
frame("Input","Vowels in IceCreAm: I, e, e, A.",[row("characters","IceCreAm",[0,2,5,6])],[3,4,5]),
frame("Move right pointer","right=7 is m, not a vowel. Move to right=6 (A).",[row("characters","IceCreAm",[0,6])],[7,8]),
frame("Swap I and A","Swap at indexes 0 and 6; move pointers inward.",[row("characters","AceCreIm",[0,6])],[9,10]),
frame("Find inner vowels","Skip consonant c; find e at index 2 and e at index 5.",[row("characters","AceCreIm",[2,5])],[7,8]),
frame("Finish","Swapping e and e makes no change. Return modified string.",[row("result","AceCreIm",[0,2,5,6])],[9,10,11])
],"Linear scan with two inward-moving pointers.")
]),
lesson("151","Reverse Words in a String","Strings",[
"Manual algorithm: normalize spaces, reverse the whole character list, then reverse each word.",
"deque.appendleft(word) adds each new word to the front.",
"split() without parameters removes redundant whitespace automatically.",
"Built-in version is the most concise; manual is best for showing algorithm fundamentals."
],[
variant("Manual: reverse twice",[
"class Solution:",
"    def trim_spaces(self, s):",
"        a = ' '.join(s.split())",
"        return list(a)",
"    def reverse(self, a, i, j):",
"        while i < j:",
"            a[i], a[j] = a[j], a[i]",
"            i, j = i+1, j-1",
"    def reverseWords(self, s):",
"        a = self.trim_spaces(s)",
"        self.reverse(a, 0, len(a)-1)",
"        start = 0",
"        for end in range(len(a)+1):",
"            if end == len(a) or a[end] == ' ':",
"                self.reverse(a, start, end-1)",
"                start = end+1",
"        return ''.join(a)"
],"O(n)","O(n)",[
frame("Input with spaces","Example string has leading, trailing and doubled spaces.",[row("input","  the  sky is blue  ")],[10]),
frame("Trim + compress spaces","After normalization, every word has one space between it.",[row("normalized","the sky is blue")],[2,3,4,10]),
frame("Reverse entire sequence","All characters reversed: eulb si yks eht.",[row("reversed","eulb si yks eht")],[5,6,7,8,11]),
frame("Reverse each individual word","eulb→blue, si→is, yks→sky, eht→the.",[row("result","blue is sky the")],[12,13,14,15,16,17])
],"Shows how reversing everything and then each word changes order without changing word spelling."),
variant("Deque appendleft",[
"from collections import deque",
"class Solution:",
"    def reverseWords(self, s):",
"        d = deque()",
"        for word in s.split():",
"            d.appendleft(word)",
"        return ' '.join(d)"
],"O(n)","O(n)",[
frame("Input","The words are the, sky, is, blue.",[row("words",["the","sky","is","blue"])],[3]),
frame("Append each word at the front","Deque changes as each word arrives.",[row("deque",["blue","is","sky","the"],[0])],[4,5,6]),
frame("Join","Join deque words with one space.",[row("answer","blue is sky the")],[7])
],"Deque supports fast appending to the front."),
variant("split + reversed + join",[
"class Solution:",
"    def reverseWords(self, s):",
"        return ' '.join(reversed(s.split()))"
],"O(n)","O(n)",[
frame("Input","Leading, trailing, and repeated spaces appear.",[row("input","  the  sky is blue  ")],[2]),
frame("split()","Split into words while removing extra spaces.",[row("words",["the","sky","is","blue"])],[3]),
frame("reversed()","Reverse the word order.",[row("reversed",["blue","is","sky","the"])],[3]),
frame("join()","Put exactly one space between words.",[row("answer","blue is sky the")],[3])
],"Short, idiomatic Python solution.")
],"For production Python, split + reversed + join is clearest. For an interview without built-ins, explain the manual double-reversal method."),
lesson("238","Product of Array Except Self","Prefix / Suffix",[
"L[i] multiplies all numbers strictly left of index i. R[i] multiplies all strictly right of it.",
"A product of an empty set of elements is 1 (multiplication identity).",
"The optimized version uses answer for prefix products and one variable R for running suffix.",
"Both solutions take O(n) time; optimized uses O(1) auxiliary space excluding output."
],[
variant("Separate left and right arrays",[
"class Solution:",
"    def productExceptSelf(self, nums):",
"        n = len(nums)",
"        L, R, answer = [0]*n, [0]*n, [0]*n",
"        L[0] = 1",
"        for i in range(1,n): L[i] = nums[i-1]*L[i-1]",
"        R[n-1] = 1",
"        for i in reversed(range(n-1)): R[i] = nums[i+1]*R[i+1]",
"        for i in range(n): answer[i] = L[i]*R[i]",
"        return answer"
],"O(n)","O(n) auxiliary",[
frame("Input","For each index multiply every OTHER number.",[row("nums",[1,2,3,4])],[3]),
frame("Build left products","No items left of index 0, so L[0] = 1.",[row("L",[1,1,2,6],[1,2,3])],[5,6]),
frame("Build right products","No items right of index 3, so R[3] = 1.",[row("R",[24,12,4,1],[0,1,2])],[7,8]),
frame("Multiply per index","L[i] × R[i] excludes nums[i].",[row("L",[1,1,2,6]),row("R",[24,12,4,1]),row("answer",[24,12,8,6],[0,1,2,3])],[9,10])
],"Clear visualization but allocates two additional arrays."),
variant("Running right product",[
"class Solution:",
"    def productExceptSelf(self, nums):",
"        n = len(nums)",
"        answer = [0] * n",
"        answer[0] = 1",
"        for i in range(1,n):",
"            answer[i] = nums[i-1] * answer[i-1]",
"        R = 1",
"        for i in reversed(range(n)):",
"            answer[i] = answer[i] * R",
"            R *= nums[i]",
"        return answer"
],"O(n)","O(1) auxiliary; O(n) output",[
frame("Initialize output","Input = [1,2,3,4], answer starts with zeros.",[row("nums",[1,2,3,4]),row("answer",[0,0,0,0])],[3,4]),
frame("Forward pass","Store left products directly in answer.",[row("answer",[1,1,2,6],[1,2,3])],[5,6,7]),
frame("i=3: R initially 1","answer[3]=6×1=6, then R=4.",[row("answer",[1,1,2,6],[3])],[8,9,10,11],{R:4}),
frame("i=2: R is 4","answer[2]=2×4=8, then R=12.",[row("answer",[1,1,8,6],[2])],[9,10,11],{R:12}),
frame("i=1: R is 12","answer[1]=1×12=12, then R=24.",[row("answer",[1,12,8,6],[1])],[9,10,11],{R:24}),
frame("i=0: R is 24","answer[0]=1×24=24.",[row("answer",[24,12,8,6],[0])],[9,10,12],{R:24})
],"Stores only running suffix product R, not an entire right array.")
],"Running right product wins: same linear time and constant auxiliary space rather than two O(n) helper arrays."),
lesson("334","Increasing Triplet Subsequence","Greedy",[
"Indices must increase and values must be strictly increasing, but positions need not be adjacent.",
"first_num tracks a small first threshold; second_num tracks a small possible middle threshold.",
"<= is deliberate: duplicates must not count as a strictly increasing triplet.",
"[2,1,5,0,4,6] has valid triplets (2,5,6), (1,5,6), and (0,4,6).",
"The task only asks whether a triplet exists; it does not need to return which one."
],[
variant("Greedy two thresholds",[
"class Solution:",
"    def increasingTriplet(self, nums):",
"        first_num = float('inf')",
"        second_num = float('inf')",
"        for n in nums:",
"            if n <= first_num: first_num = n",
"            elif n <= second_num: second_num = n",
"            else: return True",
"        return False"
],"O(n)","O(1)",[
frame("Input","We need indices i<j<k and nums[i]<nums[j]<nums[k].",[row("nums",[2,1,5,0,4,6])],[2,3,4]),
frame("See 2 and 1","Update first from ∞→2→1.",[row("nums",[2,1,5,0,4,6],[0,1])],[5,6],{first:1,second:"∞"}),
frame("See 5","5 is greater than first=1; set second=5.",[row("nums",[2,1,5,0,4,6],[2])],[7],{first:1,second:5}),
frame("See 0","Update first=0, while second remains 5.",[row("nums",[2,1,5,0,4,6],[3])],[6],{first:0,second:5}),
frame("See 4","4 is smaller than second=5, so second becomes 4.",[row("nums",[2,1,5,0,4,6],[3,4])],[7],{first:0,second:4}),
frame("See 6: success","6 exceeds both thresholds: triplet 0<4<6. Also 2<5<6 is valid.",[row("nums",[2,1,5,0,4,6],[3,4,5])],[8],{result:"True"})
],"Greedy keeps improving small thresholds, making a future larger third value easier to find.")
])
];
window.STUDY_GLOBAL_NOTES=[
"Think through input sizes and constraints before writing code.",
"Estimate complexity after choosing pseudocode; verify it from actual Python operations.",
"Consecutive loops add O(n)+O(n); nested loops typically multiply.",
"String concatenation, slicing, split, copying, and repetition can allocate memory.",
"Best-case speedups from early return do not necessarily change worst-case Big O.",
"Memorize patterns and reasoning rather than copying completed LeetCode solutions."
];
})();
