window.PROBLEM = {
 "id": "64",
 "title": "Minimum Path Sum",
 "level": "Medium",
 "topic": "Multidimensional DP",
 "question": "Move from the top-left to the bottom-right of a grid, only right or down. What's the smallest sum of the numbers you pass?",
 "why": "[[1,3,1],[1,5,1],[4,2,1]] → 1 → 3 → 1 → 1 → 1 = 7.",
 "idea": "Each cell = its value + the smaller of (cell above, cell to the left).",
 "input": "[[1,3,1],[1,5,1],[4,2,1]]",
 "output": "7",
 "notes": [],
 "best": "Each cell = its value + the smaller of (cell above, cell to the left).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minPathSum(self, grid):\n        m, n = len(grid), len(grid[0])\n        for i in range(m):\n            for j in range(n):\n                if i == 0 and j == 0: continue\n                up = grid[i-1][j] if i else float('inf'); left = grid[i][j-1] if j else float('inf')\n                grid[i][j] += min(up, left)\n        return grid[-1][-1]",
   "time": "O(m·n)",
   "space": "O(1)",
   "explain": "Each cell = its value + the smaller of (cell above, cell to the left).",
   "frames": [
    {
     "title": "Grid",
     "description": "Cheapest way to reach each cell.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "1",
         "3",
         "1"
        ],
        [
         "1",
         "5",
         "1"
        ],
        [
         "4",
         "2",
         "1"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 0",
     "description": "Only from the left: 1, 4, 5.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "1",
         "4",
         "5"
        ],
        [
         "1",
         "5",
         "1"
        ],
        [
         "4",
         "2",
         "1"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "0,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 1",
     "description": "1+1 = 2; 5+min(4,2) = 7; 1+min(5,7) = 6.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "1",
         "4",
         "5"
        ],
        [
         "2",
         "7",
         "6"
        ],
        [
         "4",
         "2",
         "1"
        ]
       ],
       "hot": [
        "1,0",
        "1,1",
        "1,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 2",
     "description": "4+2 = 6; 2+min(7,6) = 8; 1+min(6,8) = 7.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "1",
         "4",
         "5"
        ],
        [
         "2",
         "7",
         "6"
        ],
        [
         "6",
         "8",
         "7"
        ]
       ],
       "hot": [
        "2,0",
        "2,1",
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "Bottom-right = 7.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "1",
         "4",
         "5"
        ],
        [
         "2",
         "7",
         "6"
        ],
        [
         "6",
         "8",
         "7"
        ]
       ],
       "hot": [
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
