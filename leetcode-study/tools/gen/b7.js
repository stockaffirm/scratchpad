const {prob,R,M,F,range}=require("./lib");const E="·";
const G=(n,c)=>R(n,c,[],{},"grid");
prob({id:125,slug:"valid-palindrome",title:"Valid Palindrome",level:"Easy",topic:"Two Pointers",
x:["Ignoring spaces, punctuation and capital letters, does the text read the same forwards and backwards?","'A man, a plan, a canal: Panama' becomes amanaplanacanalpanama, which is a palindrome → True.","Two pointers from the ends moving inward; every pair must match."],
i:'"A man, a plan, a canal: Panama"',o:"True",
code:["class Solution:","    def isPalindrome(self, s):","        t = [c.lower() for c in s if c.isalnum()]","        l, r = 0, len(t) - 1","        while l < r:","            if t[l] != t[r]: return False","            l += 1; r -= 1","        return True"],
time:"O(n)",space:"O(n)",
frames:L=>{const t=[..."amanaplanacanalpanama"],f=[F("Clean","Keep only letters/digits, lower-case.",[L("t = [c.lower()")],[R("t",t)])];
 for(let k=0;k<3;k++)f.push(F("Compare ends","'"+t[k]+"' = '"+t[20-k]+"' ✓",[L("if t[l] != t[r]"),L("l += 1")],[R("t",t,[k,20-k],M({l:k,r:20-k}))]));
 f.push(F("Keep going","Every remaining pair matches until the pointers meet.",[L("while l < r")],[R("t",t,[10],M({l:10,r:10}))]));f.push(F("Return","Palindrome.",[L("return True")],[R("answer",["True"],[0])]));return f}});
prob({id:167,slug:"two-sum-ii-input-array-is-sorted",title:"Two Sum II - Input Array Is Sorted",level:"Medium",topic:"Two Pointers",
x:["The list is sorted. Find two numbers that add up to the target and return their positions (counting from 1).","[2,7,11,15], target 9: 2 + 7 = 9 → positions 1 and 2.","Pointers at both ends: sum too small → move left up; too big → move right down."],
i:"[2,7,11,15], target=9",o:"[1, 2]",
code:["class Solution:","    def twoSum(self, numbers, target):","        l, r = 0, len(numbers) - 1","        while l < r:","            s = numbers[l] + numbers[r]","            if s == target: return [l + 1, r + 1]","            if s < target: l += 1","            else: r -= 1"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[2,7,11,15],f=[F("Start","Ends of the sorted list.",[L("l, r =")],[R("numbers",a,[],M({l:0,r:3}))])];
 f.push(F("2 + 15 = 17","Too big → move r left.",[L("s ="),L("else: r -= 1")],[R("numbers",a,[0,3],M({l:0,r:3}))],{sum:17}));
 f.push(F("2 + 11 = 13","Too big → move r left.",[L("s ="),L("else: r -= 1")],[R("numbers",a,[0,2],M({l:0,r:2}))],{sum:13}));
 f.push(F("2 + 7 = 9","Equals target → positions 1 and 2.",[L("if s == target")],[R("numbers",a,[0,1],M({l:0,r:1}))],{sum:9}));return f}});
prob({id:15,slug:"3sum",title:"3Sum",level:"Medium",topic:"Two Pointers",
x:["Find all unique triples in the list that add up to 0.","[-1,0,1,2,-1,-4] → [-1,-1,2] and [-1,0,1].","Sort. Fix one number, then use two pointers on the rest; skip duplicates."],
i:"[-1,0,1,2,-1,-4]",o:"[[-1,-1,2],[-1,0,1]]",
code:["class Solution:","    def threeSum(self, nums):","        nums.sort(); res = []","        for i in range(len(nums) - 2):","            if i and nums[i] == nums[i-1]: continue","            l, r = i + 1, len(nums) - 1","            while l < r:","                s = nums[i] + nums[l] + nums[r]","                if s < 0: l += 1","                elif s > 0: r -= 1","                else:","                    res.append([nums[i], nums[l], nums[r]]); l += 1","                    while l < r and nums[l] == nums[l-1]: l += 1","        return res"],
time:"O(n²)",space:"O(1)",
frames:L=>{const a=[-4,-1,-1,0,1,2];return [F("Sorted","Fix i, then squeeze l and r.",[L("nums.sort()")],[R("nums",a)]),
F("i = -4","-4 + (-1) + 2 = -3 < 0 → l moves right … no triple works with -4.",[L("if s < 0")],[R("nums",a,[0,1,5],M({i:0,l:1,r:5}))],{sum:-3}),
F("i = -1 (first)","-1 + (-1) + 2 = 0 → save [-1,-1,2].",[L("res.append")],[R("nums",a,[1,2,5],M({i:1,l:2,r:5}))],{sum:0}),
F("Next","-1 + 0 + 1 = 0 → save [-1,0,1].",[L("res.append")],[R("nums",a,[1,3,4],M({i:1,l:3,r:4}))],{sum:0}),
F("i = -1 (second)","Same as previous i → skipped (no duplicates).",[L("if i and nums[i]")],[R("nums",a,[2],M({i:2}))]),
F("Return","Two unique triples.",[L("return res")],[R("res",["[-1,-1,2]","[-1,0,1]"],[0,1])])]}});
prob({id:209,slug:"minimum-size-subarray-sum",title:"Minimum Size Subarray Sum",level:"Medium",topic:"Sliding Window",
x:["Find the shortest run of numbers in a row whose sum is at least target.","target 7 on [2,3,1,2,4,3]: the run [4,3] adds to 7 and has length 2.","Grow the window on the right; while the sum is big enough, shrink from the left and record the length."],
i:"target=7, [2,3,1,2,4,3]",o:"2",
code:["class Solution:","    def minSubArrayLen(self, target, nums):","        l = total = 0; best = float('inf')","        for r, x in enumerate(nums):","            total += x","            while total >= target:","                best = min(best, r - l + 1); total -= nums[l]; l += 1","        return 0 if best == float('inf') else best"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[2,3,1,2,4,3];let l=0,t=0,b=Infinity;const f=[F("Start","Window empty.",[L("l = total")],[R("nums",a)])];
 a.forEach((x,r)=>{t+=x;let d="sum = "+t;while(t>=7){b=Math.min(b,r-l+1);t-=a[l];l++;d+=" ≥ 7 → record length "+b+", shrink"}f.push(F("r = "+r,d,[L("total += x"),L("while total >= target")],[R("nums",a,range(l,r),M({l,r}))],{sum:t,best:b===Infinity?"∞":b}))});
 f.push(F("Return","Shortest = "+b+".",[L("return 0 if")],[R("answer",[b],[0])]));return f}});
prob({id:3,slug:"longest-substring-without-repeating-characters",title:"Longest Substring Without Repeating Characters",level:"Medium",topic:"Sliding Window",
x:["Find the longest piece of the text with no repeated letter.","'abcabcbb': 'abc' has no repeats and length 3 → 3.","Slide a window; when a letter repeats, jump the left edge past its previous spot."],
i:'"abcabcbb"',o:"3",
code:["class Solution:","    def lengthOfLongestSubstring(self, s):","        seen, l, best = {}, 0, 0","        for r, ch in enumerate(s):","            if ch in seen and seen[ch] >= l: l = seen[ch] + 1","            seen[ch] = r; best = max(best, r - l + 1)","        return best"],
time:"O(n)",space:"O(k)",
frames:L=>{const s="abcabcbb",seen={};let l=0,b=0;const f=[F("Start","Window empty.",[L("seen, l, best")],[R("s",s)])];
 [...s].forEach((c,r)=>{let d="New letter — extend.";if(c in seen&&seen[c]>=l){l=seen[c]+1;d="'"+c+"' repeats → move left past its old spot.";}seen[c]=r;b=Math.max(b,r-l+1);f.push(F("r = "+r+" '"+c+"'",d,[L("if ch in seen"),L("seen[ch] = r")],[R("s",s,range(l,r),M({l,r}))],{best:b}))});
 f.push(F("Return","Longest = "+b+".",[L("return best")],[R("answer",[b],[0])]));return f}});
prob({id:30,slug:"substring-with-concatenation-of-all-words",title:"Substring with Concatenation of All Words",level:"Hard",topic:"Sliding Window",
x:["Find every start position where the text contains all the given words (same length), joined in any order with nothing in between.","s = 'barfoothefoobarman', words foo + bar: 'barfoo' starts at 0, 'foobar' starts at 9 → [0, 9].","Slice the text into word-sized chunks from each start and compare their counts with the word counts."],
i:'s="barfoothefoobarman", words=["foo","bar"]',o:"[0, 9]",
code:["class Solution:","    def findSubstring(self, s, words):","        w, n = len(words[0]), len(words); need = Counter(words); res = []","        for i in range(len(s) - w * n + 1):","            seen = Counter(s[i + j*w : i + (j+1)*w] for j in range(n))","            if seen == need: res.append(i)","        return res"],
time:"O(len·n·w)",space:"O(n)",
frames:L=>{const s="barfoothefoobarman";return [F("Setup","Words have length 3; the target window is 6 characters.",[L("w, n =")],[R("s",s),R("need",["foo:1","bar:1"])]),
F("i = 0","chunks: 'bar','foo' → same counts → save 0.",[L("seen = Counter"),L("if seen == need")],[R("s",s,range(0,5),M({i:0})),R("chunks",["bar","foo"],[0,1])]),
F("i = 1","chunks: 'arf','oot' → no match.",[L("if seen == need")],[R("s",s,range(1,6),M({i:1})),R("chunks",["arf","oot"],[])]),
F("i = 9","chunks: 'foo','bar' → same counts → save 9.",[L("res.append")],[R("s",s,range(9,14),M({i:9})),R("chunks",["foo","bar"],[0,1])]),
F("Return","[0, 9]",[L("return res")],[R("answer",[0,9],[0,1])])]}});
prob({id:76,slug:"minimum-window-substring",title:"Minimum Window Substring",level:"Hard",topic:"Sliding Window",
x:["Find the shortest piece of s that contains every letter of t (with repeats).","s = ADOBECODEBANC, t = ABC → the shortest piece is BANC.","Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest."],
i:'s="ADOBECODEBANC", t="ABC"',o:'"BANC"',
code:["class Solution:","    def minWindow(self, s, t):","        need, missing = Counter(t), len(t)","        l = 0; best = (0, float('inf'))","        for r, ch in enumerate(s):","            if need[ch] > 0: missing -= 1","            need[ch] -= 1","            while missing == 0:","                if r - l < best[1] - best[0]: best = (l, r)","                need[s[l]] += 1","                if need[s[l]] > 0: missing += 1","                l += 1","        return '' if best[1] == float('inf') else s[best[0]:best[1]+1]"],
time:"O(n+m)",space:"O(k)",
frames:L=>{const s="ADOBECODEBANC",f=[F("Setup","Need A, B, C.",[L("need, missing")],[R("s",s),R("need",["A","B","C"])])];const need={A:1,B:1,C:1};let miss=3,l=0,best=[0,Infinity];
 [...s].forEach((c,r)=>{if((need[c]||0)>0)miss--;need[c]=(need[c]||0)-1;let ev=false;while(miss===0){if(r-l<best[1]-best[0]){best=[l,r];ev=true}need[s[l]]++;if(need[s[l]]>0)miss++;l++}if(ev)f.push(F("Window found at r = "+r,"Valid window '"+s.slice(best[0],best[1]+1)+"' (length "+(best[1]-best[0]+1)+") — best so far.",[L("if r - l < best"),L("l += 1")],[R("s",s,range(best[0],best[1]),M({l:best[0],r:best[1]}))]))});
 f.push(F("Return","Smallest window: '"+s.slice(best[0],best[1]+1)+"'.",[L("return ''")],[R("s",s,range(best[0],best[1]),M({l:best[0],r:best[1]}))]));return f}});
const SB=[["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]];
prob({id:36,slug:"valid-sudoku",title:"Valid Sudoku",level:"Medium",topic:"Matrix",
x:["Is a partly filled 9×9 Sudoku board valid so far? No repeated digit in any row, column or 3×3 box (empty cells ignored).","The sample board has no repeats anywhere → True.","Walk every cell and remember (row, digit), (column, digit), (box, digit) in a set; a repeat means invalid."],
i:"9×9 board",o:"True",
code:["class Solution:","    def isValidSudoku(self, board):","        seen = set()","        for r in range(9):","            for c in range(9):","                v = board[r][c]","                if v == '.': continue","                keys = [('r', r, v), ('c', c, v), ('b', r // 3, c // 3, v)]","                if any(k in seen for k in keys): return False","                seen.update(keys)","        return True"],
time:"O(81)",space:"O(81)",
frames:L=>[F("Board","Check rows, columns and 3×3 boxes together.",[L("seen = set()")],[G("board",SB)]),
F("Row 0","5, 3, 7 — all different.",[L("keys ="),L("seen.update")],[R("board",SB,["0,0","0,1","0,4"],{},"grid")]),
F("Column 0","5, 6, 8, 4, 7 — all different.",[L("keys =")],[R("board",SB,["0,0","1,0","3,0","4,0","5,0"],{},"grid")]),
F("Top-left box","5, 3, 6, 9, 8 — all different.",[L("keys =")],[R("board",SB,["0,0","0,1","1,0","2,1","2,2"],{},"grid")]),
F("Return","No repeat found.",[L("return True")],[R("answer",["True"],[0])])]});
prob({id:54,slug:"spiral-matrix",title:"Spiral Matrix",level:"Medium",topic:"Matrix",
x:["Read a grid in spiral order: across the top, down the right side, back along the bottom, up the left side, then inward.","[[1,2,3],[4,5,6],[7,8,9]] → 1,2,3,6,9,8,7,4,5.","Peel off the top row, then rotate what's left so the next side becomes the top."],
i:"[[1,2,3],[4,5,6],[7,8,9]]",o:"[1,2,3,6,9,8,7,4,5]",
code:["class Solution:","    def spiralOrder(self, m):","        res = []","        while m:","            res += m.pop(0)","            m = [list(r) for r in zip(*m)][::-1]","        return res"],
time:"O(n·m)",space:"O(n·m)",
frames:L=>[F("Start","Take the first row.",[L("res += m.pop(0)")],[R("m",[[1,2,3],[4,5,6],[7,8,9]],["0,0","0,1","0,2"],{},"grid"),R("res",[E])]),
F("Rotate","Left over: [[4,5,6],[7,8,9]] rotated so the right side is now the top.",[L("zip(*m)")],[R("m",[[6,9],[5,8],[4,7]],["0,0","0,1"],{},"grid"),R("res",[1,2,3],[0,1,2])]),
F("Take 6, 9","Peel the top row.",[L("res += m.pop(0)")],[R("m",[[5,8],[4,7]],["0,0","0,1"],{},"grid"),R("res",[1,2,3,6,9],[3,4])]),
F("Rotate, take 8, 7","Rotate again; the top row is 8, 7.",[L("zip(*m)")],[R("m",[[5,4]],[],{},"grid"),R("res",[1,2,3,6,9,8,7],[5,6])]),
F("Take 4, then 5","The last pieces.",[L("res += m.pop(0)")],[R("res",[1,2,3,6,9,8,7,4,5],[7,8])])]});
prob({id:48,slug:"rotate-image",title:"Rotate Image",level:"Medium",topic:"Matrix",
x:["Rotate a square grid 90° clockwise, in place.","[[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]].","Transpose (swap rows and columns), then reverse each row."],
i:"[[1,2,3],[4,5,6],[7,8,9]]",o:"[[7,4,1],[8,5,2],[9,6,3]]",
code:["class Solution:","    def rotate(self, m):","        n = len(m)","        for i in range(n):","            for j in range(i + 1, n):","                m[i][j], m[j][i] = m[j][i], m[i][j]","        for row in m: row.reverse()"],
time:"O(n²)",space:"O(1)",
frames:L=>[F("Start","3×3 grid.",[L("n = len(m)")],[G("m",[[1,2,3],[4,5,6],[7,8,9]])]),F("Transpose","Swap m[i][j] with m[j][i].",[L("m[i][j], m[j][i]")],[R("m",[[1,4,7],[2,5,8],[3,6,9]],["0,1","1,0","0,2","2,0","1,2","2,1"],{},"grid")]),F("Reverse each row","Flip every row left↔right.",[L("row.reverse()")],[R("m",[[7,4,1],[8,5,2],[9,6,3]],["0,0","1,0","2,0"],{},"grid")])]});
prob({id:73,slug:"set-matrix-zeroes",title:"Set Matrix Zeroes",level:"Medium",topic:"Matrix",
x:["If a cell is 0, turn its whole row and column to 0.","[[1,1,1],[1,0,1],[1,1,1]] → the middle row and column become 0.","First note which rows and columns have a 0, then zero them all."],
i:"[[1,1,1],[1,0,1],[1,1,1]]",o:"[[1,0,1],[0,0,0],[1,0,1]]",
code:["class Solution:","    def setZeroes(self, m):","        rows = {i for i, r in enumerate(m) if 0 in r}","        cols = {j for r in m for j, v in enumerate(r) if v == 0}","        for i in range(len(m)):","            for j in range(len(m[0])):","                if i in rows or j in cols: m[i][j] = 0"],
time:"O(n·m)",space:"O(n+m)",
frames:L=>[F("Start","One zero in the middle.",[L("def setZeroes")],[R("m",[[1,1,1],[1,0,1],[1,1,1]],["1,1"],{},"grid")]),F("Note rows & cols","rows = {1}, cols = {1}.",[L("rows ="),L("cols =")],[R("m",[[1,1,1],[1,0,1],[1,1,1]],["1,0","1,1","1,2","0,1","2,1"],{},"grid")]),F("Zero them","Every cell in row 1 or column 1 becomes 0.",[L("if i in rows or j in cols")],[R("m",[[1,0,1],[0,0,0],[1,0,1]],["0,1","1,0","1,1","1,2","2,1"],{},"grid")])]});
prob({id:289,slug:"game-of-life",title:"Game of Life",level:"Medium",topic:"Matrix",
x:["Each cell is alive (1) or dead (0). All cells update at once: a live cell survives with 2–3 live neighbours; a dead cell comes alive with exactly 3; everything else dies or stays dead.","The 4×3 sample board becomes the next generation shown on the right.","Compute the next board from the old one (so updates don't affect each other), then copy it back."],
i:"[[0,1,0],[0,0,1],[1,1,1],[0,0,0]]",o:"[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]",
code:["class Solution:","    def gameOfLife(self, board):","        m, n = len(board), len(board[0])","        nxt = [[0] * n for _ in range(m)]","        for i in range(m):","            for j in range(n):","                live = sum(board[a][b] for a in range(max(0, i-1), min(m, i+2)) for b in range(max(0, j-1), min(n, j+2))) - board[i][j]","                nxt[i][j] = 1 if live == 3 or (board[i][j] and live == 2) else 0","        board[:] = nxt"],
time:"O(m·n)",space:"O(m·n)",
frames:L=>[F("Board","1 = alive.",[L("m, n =")],[R("board",[[0,1,0],[0,0,1],[1,1,1],[0,0,0]],[],{},"grid")]),F("Count neighbours","Cell (1,0) is dead with 3 live neighbours → becomes alive.",[L("live ="),L("nxt[i][j] =")],[R("board",[[0,1,0],[0,0,1],[1,1,1],[0,0,0]],["1,0","0,1","1,2","2,0"],{},"grid")]),F("Cell (0,1)","Alive with only 1 live neighbour → dies.",[L("nxt[i][j] =")],[R("board",[[0,1,0],[0,0,1],[1,1,1],[0,0,0]],["0,1"],{},"grid")]),F("Next generation","Every cell updated using the OLD board.",[L("board[:] = nxt")],[R("board",[[0,0,0],[1,0,1],[0,1,1],[0,1,0]],[],{},"grid")])]});
prob({id:383,slug:"ransom-note",title:"Ransom Note",level:"Easy",topic:"Hashmap",
x:["Can you build the ransom note using letters cut from the magazine (each letter used once)?","note 'aa', magazine 'aab' → the magazine has two a's → True.","Count letters in both; the note's counts must fit inside the magazine's."],
i:'ransomNote="aa", magazine="aab"',o:"True",
code:["class Solution:","    def canConstruct(self, ransomNote, magazine):","        return not (Counter(ransomNote) - Counter(magazine))"],
time:"O(n+m)",space:"O(1)",
frames:L=>[F("Count note","a:2",[L("Counter(ransomNote)")],[R("note",["a:2"])]),F("Count magazine","a:2, b:1",[L("Counter(magazine)")],[R("magazine",["a:2","b:1"])]),F("Subtract","note − magazine = nothing left over.",[L("not (")],[R("left over",[E])]),F("Return","True",[L("return not")],[R("answer",["True"],[0])])]});
prob({id:205,slug:"isomorphic-strings",title:"Isomorphic Strings",level:"Easy",topic:"Hashmap",
x:["Can you swap letters in s to get t, with each letter always mapping to the same letter (and no two letters sharing a target)?","egg → add: e→a, g→d works → True.","Keep two maps (s→t and t→s); any conflict means False."],
i:'s="egg", t="add"',o:"True",
code:["class Solution:","    def isIsomorphic(self, s, t):","        m1, m2 = {}, {}","        for a, b in zip(s, t):","            if m1.setdefault(a, b) != b or m2.setdefault(b, a) != a: return False","        return True"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Start","Compare letter by letter.",[L("m1, m2")],[R("s","egg"),R("t","add")]),F("e → a","New pair.",[L("setdefault")],[R("s","egg",[0]),R("t","add",[0]),R("map",["e→a"])]),F("g → d","New pair.",[L("setdefault")],[R("s","egg",[1]),R("t","add",[1]),R("map",["e→a","g→d"])]),F("g → d again","Same as before → consistent.",[L("setdefault")],[R("s","egg",[2]),R("t","add",[2]),R("map",["e→a","g→d"])]),F("Return","No conflicts → True.",[L("return True")],[R("answer",["True"],[0])])]});
prob({id:290,slug:"word-pattern",title:"Word Pattern",level:"Easy",topic:"Hashmap",
x:["Does the text follow the pattern, with each pattern letter standing for exactly one word (and vice versa)?","pattern abba and 'dog cat cat dog': a=dog, b=cat → True.","Two maps again: letter→word and word→letter."],
i:'pattern="abba", s="dog cat cat dog"',o:"True",
code:["class Solution:","    def wordPattern(self, pattern, s):","        w = s.split()","        if len(w) != len(pattern): return False","        m1, m2 = {}, {}","        for p, x in zip(pattern, w):","            if m1.setdefault(p, x) != x or m2.setdefault(x, p) != p: return False","        return True"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Start","Split into words.",[L("w = s.split()")],[R("pattern",["a","b","b","a"]),R("words",["dog","cat","cat","dog"])]),F("a ↔ dog","New pair.",[L("setdefault")],[R("pattern",["a","b","b","a"],[0]),R("words",["dog","cat","cat","dog"],[0]),R("map",["a↔dog"])]),F("b ↔ cat","New pair.",[L("setdefault")],[R("pattern",["a","b","b","a"],[1]),R("words",["dog","cat","cat","dog"],[1]),R("map",["a↔dog","b↔cat"])]),F("b ↔ cat, a ↔ dog","Repeats match their earlier pairs.",[L("setdefault")],[R("pattern",["a","b","b","a"],[2,3]),R("words",["dog","cat","cat","dog"],[2,3]),R("map",["a↔dog","b↔cat"])]),F("Return","True",[L("return True")],[R("answer",["True"],[0])])]});
prob({id:242,slug:"valid-anagram",title:"Valid Anagram",level:"Easy",topic:"Hashmap",
x:["Are the two words made of exactly the same letters (rearranged)?","'anagram' and 'nagaram' both have a×3, n×1, g×1, r×1, m×1 → True.","Count the letters in each and compare."],
i:'s="anagram", t="nagaram"',o:"True",
code:["class Solution:","    def isAnagram(self, s, t):","        return Counter(s) == Counter(t)"],
time:"O(n)",space:"O(1)",
frames:L=>[F("Count s","a:3 n:1 g:1 r:1 m:1",[L("Counter(s)")],[R("s","anagram"),R("counts",["a:3","n:1","g:1","r:1","m:1"])]),F("Count t","a:3 n:1 g:1 r:1 m:1",[L("Counter(t)")],[R("t","nagaram"),R("counts",["a:3","n:1","g:1","r:1","m:1"])]),F("Compare","Equal → True.",[L("==")],[R("answer",["True"],[0])])]});
prob({id:49,slug:"group-anagrams",title:"Group Anagrams",level:"Medium",topic:"Hashmap",
x:["Group together the words that are anagrams of each other.","eat, tea, ate share letters a-e-t; tan, nat share a-n-t; bat is alone.","Sort each word's letters to get a key; words with the same key go in the same group."],
i:'["eat","tea","tan","ate","nat","bat"]',o:'[["eat","tea","ate"],["tan","nat"],["bat"]]',
code:["class Solution:","    def groupAnagrams(self, strs):","        groups = defaultdict(list)","        for w in strs:","            groups[''.join(sorted(w))].append(w)","        return list(groups.values())"],
time:"O(n·k log k)",space:"O(n·k)",
frames:L=>{const w=["eat","tea","tan","ate","nat","bat"],g={},f=[F("Start","Key = sorted letters.",[L("groups =")],[R("words",w)])];
 w.forEach((x,i)=>{const k=[...x].sort().join("");(g[k]=g[k]||[]).push(x);f.push(F("'"+x+"' → key '"+k+"'","Add to group '"+k+"'.",[L("groups[''.join")],[R("words",w,[i]),...Object.entries(g).map(([a,b])=>R("'"+a+"'",b))]))});
 f.push(F("Return","3 groups.",[L("return list")],[R("answer",["[eat,tea,ate]","[tan,nat]","[bat]"],[0,1,2])]));return f}});
prob({id:1,slug:"two-sum",title:"Two Sum",level:"Easy",topic:"Hashmap",
x:["Find two numbers in the list that add up to the target and return their positions.","[2,7,11,15], target 9: 2 + 7 = 9 → positions 0 and 1.","For each number, check if (target − number) was already seen, using a dictionary."],
i:"[2,7,11,15], target=9",o:"[0, 1]",
code:["class Solution:","    def twoSum(self, nums, target):","        seen = {}","        for i, x in enumerate(nums):","            if target - x in seen: return [seen[target - x], i]","            seen[x] = i"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Start","seen is empty.",[L("seen = {}")],[R("nums",[2,7,11,15])]),F("x = 2","Need 7 — not seen yet. Remember 2.",[L("if target - x in seen"),L("seen[x] = i")],[R("nums",[2,7,11,15],[0]),R("seen",["2→0"])]),F("x = 7","Need 2 — seen at index 0!",[L("return [seen")],[R("nums",[2,7,11,15],[0,1]),R("seen",["2→0"])]),F("Return","[0, 1]",[L("return [seen")],[R("answer",[0,1],[0,1])])]});
prob({id:202,slug:"happy-number",title:"Happy Number",level:"Easy",topic:"Hashmap",
x:["Repeatedly replace a number by the sum of the squares of its digits. If it reaches 1 it's happy; if it loops forever it isn't.","19 → 82 → 68 → 100 → 1 → happy.","Remember numbers you've seen; seeing one again means a loop."],
i:"19",o:"True",
code:["class Solution:","    def isHappy(self, n):","        seen = set()","        while n != 1 and n not in seen:","            seen.add(n)","            n = sum(int(d) ** 2 for d in str(n))","        return n == 1"],
time:"O(log n)",space:"O(log n)",
frames:L=>{const f=[F("Start","n = 19",[L("seen = set()")],[R("n",[19])])];let n=19;const path=[19];
 while(n!==1){const nx=[...String(n)].reduce((s,d)=>s+d*d,0);f.push(F(n+" →",[...String(n)].join("² + ")+"² = "+nx,[L("n = sum")],[R("path",path.concat(nx),[path.length],{},"list")]));path.push(nx);n=nx}
 f.push(F("Return","Reached 1 → happy.",[L("return n == 1")],[R("answer",["True"],[0])]));return f}});
prob({id:219,slug:"contains-duplicate-ii",title:"Contains Duplicate II",level:"Easy",topic:"Hashmap",
x:["Are there two equal numbers whose positions differ by at most k?","[1,2,3,1], k = 3: the two 1s are 3 apart → True.","Remember each number's last position in a dictionary."],
i:"[1,2,3,1], k=3",o:"True",
code:["class Solution:","    def containsNearbyDuplicate(self, nums, k):","        last = {}","        for i, x in enumerate(nums):","            if x in last and i - last[x] <= k: return True","            last[x] = i","        return False"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Start","Track the last index of each value.",[L("last = {}")],[R("nums",[1,2,3,1])]),F("1, 2, 3","All new → record positions.",[L("last[x] = i")],[R("nums",[1,2,3,1],[0,1,2]),R("last",["1→0","2→1","3→2"])]),F("Second 1","Last seen at 0; distance 3 − 0 = 3 ≤ k → True.",[L("if x in last")],[R("nums",[1,2,3,1],[0,3]),R("last",["1→0","2→1","3→2"])])]});
prob({id:128,slug:"longest-consecutive-sequence",title:"Longest Consecutive Sequence",level:"Medium",topic:"Hashmap",
x:["Find the longest run of consecutive integers (like 1,2,3,4) in the unsorted list, in O(n).","[100,4,200,1,3,2] contains 1,2,3,4 → length 4.","Put numbers in a set. Only start counting from numbers with no smaller neighbour."],
i:"[100,4,200,1,3,2]",o:"4",
code:["class Solution:","    def longestConsecutive(self, nums):","        s, best = set(nums), 0","        for x in s:","            if x - 1 not in s:","                y = x","                while y + 1 in s: y += 1","                best = max(best, y - x + 1)","        return best"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Set","Fast lookups.",[L("s, best")],[R("set",[100,4,200,1,3,2])]),F("Skip 4, 3, 2","They have a smaller neighbour — not the start of a run.",[L("if x - 1 not in s")],[R("set",[100,4,200,1,3,2],[1,4,5])]),F("Start at 1","1 has no 0 → run 1,2,3,4 → length 4.",[L("while y + 1 in s")],[R("run",[1,2,3,4],[0,1,2,3],{},"list")],{best:4}),F("Starts 100, 200","Runs of length 1.",[L("best = max")],[R("set",[100,4,200,1,3,2],[0,2])],{best:4}),F("Return","4",[L("return best")],[R("answer",[4],[0])])]});
prob({id:228,slug:"summary-ranges",title:"Summary Ranges",level:"Easy",topic:"Intervals",
x:["Turn a sorted list of unique numbers into the shortest list of ranges.","[0,1,2,4,5,7] → 0→2, 4→5, 7.","Walk along; extend each run while the next number is exactly one bigger."],
i:"[0,1,2,4,5,7]",o:'["0->2","4->5","7"]',
code:["class Solution:","    def summaryRanges(self, nums):","        res, i = [], 0","        while i < len(nums):","            j = i","            while j + 1 < len(nums) and nums[j+1] == nums[j] + 1: j += 1","            res.append(str(nums[i]) if i == j else f'{nums[i]}->{nums[j]}')","            i = j + 1","        return res"],
time:"O(n)",space:"O(1)",
frames:L=>{const a=[0,1,2,4,5,7],f=[F("Start","Find the end of each run.",[L("res, i")],[R("nums",a)])];
 [[0,2],[3,4],[5,5]].forEach(([i,j])=>f.push(F("Run "+a[i]+"…"+a[j],"Record "+(i===j?a[i]:a[i]+"->"+a[j]),[L("while j + 1"),L("res.append")],[R("nums",a,range(i,j),M({i,j}))])));
 f.push(F("Return",'["0->2","4->5","7"]',[L("return res")],[R("res",["0->2","4->5","7"],[0,1,2])]));return f}});
prob({id:56,slug:"merge-intervals",title:"Merge Intervals",level:"Medium",topic:"Intervals",
x:["Merge all overlapping ranges.","[1,3] and [2,6] overlap → [1,6]; [8,10] and [15,18] stand alone.","Sort by start. If the next one starts before the last one ends, extend the last one."],
i:"[[1,3],[2,6],[8,10],[15,18]]",o:"[[1,6],[8,10],[15,18]]",
code:["class Solution:","    def merge(self, intervals):","        intervals.sort()","        res = [intervals[0]]","        for s, e in intervals[1:]:","            if s <= res[-1][1]: res[-1][1] = max(res[-1][1], e)","            else: res.append([s, e])","        return res"],
time:"O(n log n)",space:"O(n)",
frames:L=>[F("Sorted","Already sorted by start.",[L("intervals.sort()")],[R("intervals",["[1,3]","[2,6]","[8,10]","[15,18]"]),R("res",["[1,3]"],[0])]),F("[2,6]","2 ≤ 3 → overlap → [1,6].",[L("if s <= res[-1][1]")],[R("intervals",["[1,3]","[2,6]","[8,10]","[15,18]"],[1]),R("res",["[1,6]"],[0])]),F("[8,10]","8 > 6 → new interval.",[L("else: res.append")],[R("intervals",["[1,3]","[2,6]","[8,10]","[15,18]"],[2]),R("res",["[1,6]","[8,10]"],[1])]),F("[15,18]","15 > 10 → new interval.",[L("else: res.append")],[R("intervals",["[1,3]","[2,6]","[8,10]","[15,18]"],[3]),R("res",["[1,6]","[8,10]","[15,18]"],[2])])]});
prob({id:57,slug:"insert-interval",title:"Insert Interval",level:"Medium",topic:"Intervals",
x:["A sorted list of non-overlapping ranges plus one new range. Insert it and merge anything that overlaps.","Insert [4,8] into [1,2],[3,5],[6,7],[8,10],[12,16] → it swallows [3,5],[6,7],[8,10] → [3,10].","Three phases: copy ranges that end before the new one, merge the overlapping ones, copy the rest."],
i:"[[1,2],[3,5],[6,7],[8,10],[12,16]], new=[4,8]",o:"[[1,2],[3,10],[12,16]]",
code:["class Solution:","    def insert(self, intervals, new):","        res, i, n = [], 0, len(intervals)","        while i < n and intervals[i][1] < new[0]:","            res.append(intervals[i]); i += 1","        while i < n and intervals[i][0] <= new[1]:","            new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]; i += 1","        res.append(new)","        res += intervals[i:]","        return res"],
time:"O(n)",space:"O(n)",
frames:L=>[F("Start","New range [4,8].",[L("res, i, n")],[R("intervals",["[1,2]","[3,5]","[6,7]","[8,10]","[12,16]"])]),F("Before","[1,2] ends before 4 → copy as is.",[L("while i < n and intervals[i][1] < new[0]")],[R("intervals",["[1,2]","[3,5]","[6,7]","[8,10]","[12,16]"],[0]),R("res",["[1,2]"],[0])]),F("Merge","[3,5], [6,7], [8,10] all overlap [4,8] → [3,10].",[L("new = [min")],[R("intervals",["[1,2]","[3,5]","[6,7]","[8,10]","[12,16]"],[1,2,3]),R("new",["[3,10]"],[0])]),F("After","Add merged range, then copy [12,16].",[L("res.append(new)"),L("res += intervals[i:]")],[R("res",["[1,2]","[3,10]","[12,16]"],[1])])]});
