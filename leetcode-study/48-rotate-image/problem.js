window.PROBLEM = {
 "id": "48",
 "title": "Rotate Image",
 "level": "Medium",
 "topic": "Matrix",
 "question": "Rotate a square grid 90° clockwise, in place.",
 "why": "[[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]].",
 "idea": "Transpose (swap rows and columns), then reverse each row.",
 "input": "[[1,2,3],[4,5,6],[7,8,9]]",
 "output": "[[7,4,1],[8,5,2],[9,6,3]]",
 "notes": [],
 "best": "Transpose (swap rows and columns), then reverse each row.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def rotate(self, m):\n        n = len(m)\n        for i in range(n):\n            for j in range(i + 1, n):\n                m[i][j], m[j][i] = m[j][i], m[i][j]\n        for row in m: row.reverse()",
   "time": "O(n²)",
   "space": "O(1)",
   "explain": "Transpose (swap rows and columns), then reverse each row.",
   "frames": [
    {
     "title": "Start",
     "description": "3×3 grid.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "1",
         "2",
         "3"
        ],
        [
         "4",
         "5",
         "6"
        ],
        [
         "7",
         "8",
         "9"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Transpose",
     "description": "Swap m[i][j] with m[j][i].",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "1",
         "4",
         "7"
        ],
        [
         "2",
         "5",
         "8"
        ],
        [
         "3",
         "6",
         "9"
        ]
       ],
       "hot": [
        "0,1",
        "1,0",
        "0,2",
        "2,0",
        "1,2",
        "2,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Reverse each row",
     "description": "Flip every row left↔right.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "7",
         "4",
         "1"
        ],
        [
         "8",
         "5",
         "2"
        ],
        [
         "9",
         "6",
         "3"
        ]
       ],
       "hot": [
        "0,0",
        "1,0",
        "2,0"
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
