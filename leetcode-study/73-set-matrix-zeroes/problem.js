window.PROBLEM = {
 "id": "73",
 "title": "Set Matrix Zeroes",
 "level": "Medium",
 "topic": "Matrix",
 "question": "If a cell is 0, turn its whole row and column to 0.",
 "why": "[[1,1,1],[1,0,1],[1,1,1]] → the middle row and column become 0.",
 "idea": "First note which rows and columns have a 0, then zero them all.",
 "input": "[[1,1,1],[1,0,1],[1,1,1]]",
 "output": "[[1,0,1],[0,0,0],[1,0,1]]",
 "notes": [],
 "best": "First note which rows and columns have a 0, then zero them all.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def setZeroes(self, m):\n        rows = {i for i, r in enumerate(m) if 0 in r}\n        cols = {j for r in m for j, v in enumerate(r) if v == 0}\n        for i in range(len(m)):\n            for j in range(len(m[0])):\n                if i in rows or j in cols: m[i][j] = 0",
   "time": "O(n·m)",
   "space": "O(n+m)",
   "explain": "First note which rows and columns have a 0, then zero them all.",
   "frames": [
    {
     "title": "Start",
     "description": "One zero in the middle.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "1",
         "1",
         "1"
        ],
        [
         "1",
         "0",
         "1"
        ],
        [
         "1",
         "1",
         "1"
        ]
       ],
       "hot": [
        "1,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Note rows & cols",
     "description": "rows = {1}, cols = {1}.",
     "codeLines": [
      3,
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "1",
         "1",
         "1"
        ],
        [
         "1",
         "0",
         "1"
        ],
        [
         "1",
         "1",
         "1"
        ]
       ],
       "hot": [
        "1,0",
        "1,1",
        "1,2",
        "0,1",
        "2,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Zero them",
     "description": "Every cell in row 1 or column 1 becomes 0.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "1",
         "0",
         "1"
        ],
        [
         "0",
         "0",
         "0"
        ],
        [
         "1",
         "0",
         "1"
        ]
       ],
       "hot": [
        "0,1",
        "1,0",
        "1,1",
        "1,2",
        "2,1"
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
