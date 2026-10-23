window.PROBLEM = {
 "id": "221",
 "title": "Maximal Square",
 "level": "Medium",
 "topic": "Multidimensional DP",
 "question": "In a grid of 0s and 1s, find the largest square made only of 1s and return its AREA.",
 "why": "The largest all-1 square in the sample is 2×2 → area 4.",
 "idea": "dp[i][j] = side of the biggest square ending at that cell = 1 + min(up, left, up-left).",
 "input": "4×5 grid of 1s and 0s",
 "output": "4",
 "notes": [],
 "best": "dp[i][j] = side of the biggest square ending at that cell = 1 + min(up, left, up-left).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maximalSquare(self, m):\n        rows, cols = len(m), len(m[0]); dp = [[0] * (cols + 1) for _ in range(rows + 1)]; best = 0\n        for i in range(1, rows + 1):\n            for j in range(1, cols + 1):\n                if m[i-1][j-1] == '1':\n                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]); best = max(best, dp[i][j])\n        return best * best",
   "time": "O(m·n)",
   "space": "O(m·n)",
   "explain": "dp[i][j] = side of the biggest square ending at that cell = 1 + min(up, left, up-left).",
   "frames": [
    {
     "title": "Grid",
     "description": "1 = filled.",
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
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "1",
         "1",
         "1",
         "1",
         "1"
        ],
        [
         "1",
         "0",
         "0",
         "1",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 1",
     "description": "Square side ending at each cell.",
     "codeLines": [
      7
     ],
     "facts": {
      "best": 1
     },
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 2",
     "description": "Square side ending at each cell.",
     "codeLines": [
      7
     ],
     "facts": {
      "best": 1
     },
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 3",
     "description": "Square side ending at each cell.",
     "codeLines": [
      7
     ],
     "facts": {
      "best": 2
     },
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "3,4",
        "3,5"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 4",
     "description": "Square side ending at each cell.",
     "codeLines": [
      7
     ],
     "facts": {
      "best": 2
     },
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "1",
         "0",
         "0",
         "1",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "Biggest side = 2 → area 4.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "4"
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
