window.PROBLEM = {
 "id": "52",
 "title": "N-Queens II",
 "level": "Hard",
 "topic": "Backtracking",
 "question": "Place n queens on an n×n chessboard so none attack each other (no shared row, column or diagonal). How many ways?",
 "why": "On a 4×4 board there are 2 solutions.",
 "idea": "Place one queen per row; skip columns and diagonals already used; undo when stuck.",
 "input": "n = 4",
 "output": "2",
 "notes": [],
 "best": "Place one queen per row; skip columns and diagonals already used; undo when stuck.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def totalNQueens(self, n):\n        cols, d1, d2 = set(), set(), set()\n        def go(r):\n            if r == n: return 1\n            count = 0\n            for c in range(n):\n                if c in cols or r - c in d1 or r + c in d2: continue\n                cols.add(c); d1.add(r - c); d2.add(r + c)\n                count += go(r + 1)\n                cols.remove(c); d1.remove(r - c); d2.remove(r + c)\n            return count\n        return go(0)",
   "time": "O(n!)",
   "space": "O(n)",
   "explain": "Place one queen per row; skip columns and diagonals already used; undo when stuck.",
   "frames": [
    {
     "title": "Row by row",
     "description": "One queen per row, avoiding attacked squares.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "·",
         "·",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "·",
         "·"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Solution 1",
     "description": "Queens at columns 1, 3, 0, 2.",
     "codeLines": [
      5
     ],
     "facts": {
      "found": 1
     },
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "·",
         "Q",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "·",
         "Q"
        ],
        [
         "Q",
         "·",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "Q",
         "·"
        ]
       ],
       "hot": [
        "0,1",
        "1,3",
        "2,0",
        "3,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Solution 2",
     "description": "Queens at columns 2, 0, 3, 1 (the mirror image).",
     "codeLines": [
      5
     ],
     "facts": {
      "found": 2
     },
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "·",
         "·",
         "Q",
         "·"
        ],
        [
         "Q",
         "·",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "·",
         "Q"
        ],
        [
         "·",
         "Q",
         "·",
         "·"
        ]
       ],
       "hot": [
        "0,2",
        "1,0",
        "2,3",
        "3,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "2 solutions.",
     "codeLines": [
      13
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "2"
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
