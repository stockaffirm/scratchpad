window.PROBLEM = {
 "id": "76",
 "title": "Minimum Window Substring",
 "level": "Hard",
 "topic": "Sliding Window",
 "question": "Find the shortest piece of s that contains every letter of t (with repeats).",
 "why": "s = ADOBECODEBANC, t = ABC → the shortest piece is BANC.",
 "idea": "Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest.",
 "input": "s=\"ADOBECODEBANC\", t=\"ABC\"",
 "output": "\"BANC\"",
 "notes": [],
 "best": "Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minWindow(self, s, t):\n        need, missing = Counter(t), len(t)\n        l = 0; best = (0, float('inf'))\n        for r, ch in enumerate(s):\n            if need[ch] > 0: missing -= 1\n            need[ch] -= 1\n            while missing == 0:\n                if r - l < best[1] - best[0]: best = (l, r)\n                need[s[l]] += 1\n                if need[s[l]] > 0: missing += 1\n                l += 1\n        return '' if best[1] == float('inf') else s[best[0]:best[1]+1]",
   "time": "O(n+m)",
   "space": "O(k)",
   "explain": "Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest.",
   "frames": [
    {
     "title": "Setup",
     "description": "Need A, B, C.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "need",
       "cells": [
        "A",
        "B",
        "C"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Window found at r = 5",
     "description": "Valid window 'ADOBEC' (length 6) — best so far.",
     "codeLines": [
      9,
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {
        "0": "l",
        "5": "r"
       }
      }
     ]
    },
    {
     "title": "Window found at r = 12",
     "description": "Valid window 'BANC' (length 4) — best so far.",
     "codeLines": [
      9,
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [
        9,
        10,
        11,
        12
       ],
       "marks": {
        "9": "l",
        "12": "r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Smallest window: 'BANC'.",
     "codeLines": [
      13
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [
        9,
        10,
        11,
        12
       ],
       "marks": {
        "9": "l",
        "12": "r"
       }
      }
     ]
    }
   ]
  }
 ]
};
