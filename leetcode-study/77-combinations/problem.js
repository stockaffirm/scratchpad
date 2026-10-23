window.PROBLEM = {
 "id": "77",
 "title": "Combinations",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "List every way to choose k numbers from 1 to n (order doesn't matter).",
 "why": "n = 4, k = 2 → [1,2],[1,3],[1,4],[2,3],[2,4],[3,4].",
 "idea": "Backtracking: pick numbers in increasing order so you never repeat a set.",
 "input": "n = 4, k = 2",
 "output": "[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]",
 "notes": [],
 "best": "Backtracking: pick numbers in increasing order so you never repeat a set.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def combine(self, n, k):\n        res = []\n        def go(start, path):\n            if len(path) == k: res.append(path[:]); return\n            for x in range(start, n + 1):\n                path.append(x); go(x + 1, path); path.pop()\n        go(1, [])\n        return res",
   "time": "O(C(n,k)·k)",
   "space": "O(k)",
   "explain": "Backtracking: pick numbers in increasing order so you never repeat a set.",
   "frames": [
    {
     "title": "Start",
     "description": "Choose 2 numbers from 1–4.",
     "codeLines": [
      8
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
     "title": "Start with 1",
     "description": "Pair 1 with 2, 3, 4.",
     "codeLines": [
      6
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
        "[1,2]",
        "[1,3]",
        "[1,4]"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Start with 2",
     "description": "Pair 2 with 3, 4 (not 1 — avoids repeats).",
     "codeLines": [
      7
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
        "[1,2]",
        "[1,3]",
        "[1,4]",
        "[2,3]",
        "[2,4]"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Start with 3",
     "description": "Pair 3 with 4.",
     "codeLines": [
      5
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
        "…",
        "[3,4]"
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
     "description": "6 combinations.",
     "codeLines": [
      9
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
