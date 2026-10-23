window.PROBLEM = {
 "id": "216",
 "title": "Combination Sum III",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "Choose k distinct numbers from 1–9 that add up to n.",
 "input": "k = 3, n = 7",
 "output": "[[1,2,4]]",
 "notes": [
  "start = x + 1 keeps numbers distinct and ordered.",
  "break (not continue) because later x are larger."
 ],
 "best": "Backtrack in increasing order and prune once the sum exceeds n.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def combinationSum3(self, k, n):\n        res = []\n        def go(start, path, total):\n            if len(path) == k:\n                if total == n: res.append(path[:])\n                return\n            for x in range(start, 10):\n                if total + x > n: break\n                path.append(x); go(x + 1, path, total + x); path.pop()\n        go(1, [], 0)\n        return res",
   "time": "O(C(9,k)·k)",
   "space": "O(k)",
   "explain": "Increasing choices avoid duplicates; prune when over n.",
   "frames": [
    {
     "title": "Start",
     "description": "Choose 3 numbers from 1–9 summing to 7.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Build [1, 2]",
     "description": "Sum 3; need one more.",
     "codeLines": [
      10
     ],
     "facts": {
      "total": 3
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "1",
        "2"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Try 3 → [1,2,3]",
     "description": "Size 3 but sum 6 ≠ 7 → reject.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "total": 6
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Try 4 → [1,2,4]",
     "description": "Sum 7 = n → save!",
     "codeLines": [
      6
     ],
     "facts": {
      "total": 7
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "[1,2,4]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Try 5",
     "description": "3 + 5 = 8 > 7 → prune (break).",
     "codeLines": [
      9
     ],
     "facts": {
      "total": 8
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "1",
        "2",
        "5"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Every other start is too big.",
     "codeLines": [
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "[1,2,4]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
