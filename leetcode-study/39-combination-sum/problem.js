window.PROBLEM = {
 "id": "39",
 "title": "Combination Sum",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "Choose numbers from the list (each can be used any number of times) that add up to the target. List all unique combinations.",
 "why": "candidates [2,3,6,7], target 7 → [2,2,3] and [7].",
 "idea": "Backtracking: add a number, recurse (it can repeat), and stop when the sum is too big.",
 "input": "candidates=[2,3,6,7], target=7",
 "output": "[[2,2,3],[7]]",
 "notes": [],
 "best": "Backtracking: add a number, recurse (it can repeat), and stop when the sum is too big.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def combinationSum(self, c, target):\n        res = []\n        def go(start, path, total):\n            if total == target: res.append(path[:]); return\n            for i in range(start, len(c)):\n                if total + c[i] <= target:\n                    path.append(c[i]); go(i, path, total + c[i]); path.pop()\n        go(0, [], 0)\n        return res",
   "time": "O(2^t)",
   "space": "O(t)",
   "explain": "Backtracking: add a number, recurse (it can repeat), and stop when the sum is too big.",
   "frames": [
    {
     "title": "Start",
     "description": "Target 7.",
     "codeLines": [
      9
     ],
     "facts": {
      "total": 0
     },
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
     "title": "2, 2, 2",
     "description": "Sum 6 — adding another 2 would be 8 → too big.",
     "codeLines": [
      7
     ],
     "facts": {
      "total": 6
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "2",
        "2",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "2, 2, 3",
     "description": "Sum 7 = target → save.",
     "codeLines": [
      5
     ],
     "facts": {
      "total": 7
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "2",
        "2",
        "3"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "found",
       "cells": [
        "[2,2,3]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "7",
     "description": "Sum 7 = target → save.",
     "codeLines": [
      5
     ],
     "facts": {
      "total": 7
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "found",
       "cells": [
        "[2,2,3]",
        "[7]"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Two combinations.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "[2,2,3]",
        "[7]"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
