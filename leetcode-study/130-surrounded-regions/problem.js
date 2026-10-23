window.PROBLEM = {
 "id": "130",
 "title": "Surrounded Regions",
 "level": "Medium",
 "topic": "Graph - General",
 "question": "Capture every group of 'O's that is completely surrounded by 'X'. 'O's connected to the border stay.",
 "why": "The big O group is enclosed → becomes X. The bottom O touches the border → stays.",
 "idea": "Start from border 'O's and mark everything connected to them as safe. Flip the rest.",
 "input": "4×4 board of X and O",
 "output": "enclosed O's become X",
 "notes": [],
 "best": "Start from border 'O's and mark everything connected to them as safe. Flip the rest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def solve(self, board):\n        m, n = len(board), len(board[0])\n        def mark(r, c):\n            if 0 <= r < m and 0 <= c < n and board[r][c] == 'O':\n                board[r][c] = '#'\n                mark(r+1, c); mark(r-1, c); mark(r, c+1); mark(r, c-1)\n        for r in range(m): mark(r, 0); mark(r, n-1)\n        for c in range(n): mark(0, c); mark(m-1, c)\n        for r in range(m):\n            for c in range(n): board[r][c] = 'O' if board[r][c] == '#' else 'X'",
   "time": "O(m·n)",
   "space": "O(m·n)",
   "explain": "Start from border 'O's and mark everything connected to them as safe. Flip the rest.",
   "frames": [
    {
     "title": "Board",
     "description": "Which O's can escape to the border?",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "X",
         "X",
         "X",
         "X"
        ],
        [
         "X",
         "O",
         "O",
         "X"
        ],
        [
         "X",
         "X",
         "O",
         "X"
        ],
        [
         "X",
         "O",
         "X",
         "X"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Mark safe",
     "description": "Border O at (3,1) is safe → mark #.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "X",
         "X",
         "X",
         "X"
        ],
        [
         "X",
         "O",
         "O",
         "X"
        ],
        [
         "X",
         "X",
         "O",
         "X"
        ],
        [
         "X",
         "#",
         "X",
         "X"
        ]
       ],
       "hot": [
        "3,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Flip",
     "description": "Unmarked O's become X; # goes back to O.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "X",
         "X",
         "X",
         "X"
        ],
        [
         "X",
         "X",
         "X",
         "X"
        ],
        [
         "X",
         "X",
         "X",
         "X"
        ],
        [
         "X",
         "O",
         "X",
         "X"
        ]
       ],
       "hot": [
        "1,1",
        "1,2",
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    }
   ]
  }
 ]
};
