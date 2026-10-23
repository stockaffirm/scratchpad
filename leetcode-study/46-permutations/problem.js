window.PROBLEM = {
 "id": "46",
 "title": "Permutations",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "List every possible ordering of the given distinct numbers.",
 "why": "[1,2,3] has 3 × 2 × 1 = 6 orderings.",
 "idea": "Backtracking: pick an unused number for each position, then undo and try the next.",
 "input": "[1,2,3]",
 "output": "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]",
 "notes": [],
 "best": "Backtracking: pick an unused number for each position, then undo and try the next.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def permute(self, nums):\n        res = []\n        def go(path, used):\n            if len(path) == len(nums): res.append(path[:]); return\n            for i, x in enumerate(nums):\n                if i not in used:\n                    used.add(i); path.append(x); go(path, used); path.pop(); used.remove(i)\n        go([], set())\n        return res",
   "time": "O(n·n!)",
   "space": "O(n)",
   "explain": "Backtracking: pick an unused number for each position, then undo and try the next.",
   "frames": [
    {
     "title": "Start",
     "description": "Fill each position with an unused number.",
     "codeLines": [
      9
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
     "title": "First = 1",
     "description": "Then 2,3 or 3,2.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "found",
       "cells": [
        "[1,2,3]",
        "[1,3,2]"
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
     "title": "First = 2",
     "description": "Then 1,3 or 3,1.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "found",
       "cells": [
        "[2,1,3]",
        "[2,3,1]"
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
     "title": "First = 3",
     "description": "Then 1,2 or 2,1.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "found",
       "cells": [
        "[3,1,2]",
        "[3,2,1]"
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
     "title": "Return",
     "description": "6 permutations.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "count",
       "cells": [
        "6"
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
