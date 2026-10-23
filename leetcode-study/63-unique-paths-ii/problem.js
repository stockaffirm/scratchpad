window.PROBLEM = {
 "id": "63",
 "title": "Unique Paths II",
 "level": "Medium",
 "topic": "Multidimensional DP",
 "question": "Like Unique Paths, but some cells are obstacles (1) you can't step on. How many paths from top-left to bottom-right?",
 "why": "With one obstacle in the middle of a 3×3 grid there are 2 paths.",
 "idea": "Paths into a cell = paths from above + paths from the left; obstacle cells have 0.",
 "input": "[[0,0,0],[0,1,0],[0,0,0]]",
 "output": "2",
 "notes": [],
 "best": "Paths into a cell = paths from above + paths from the left; obstacle cells have 0.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def uniquePathsWithObstacles(self, g):\n        m, n = len(g), len(g[0]); dp = [0] * n; dp[0] = 1\n        for i in range(m):\n            for j in range(n):\n                if g[i][j] == 1: dp[j] = 0\n                elif j: dp[j] += dp[j - 1]\n        return dp[-1]",
   "time": "O(m·n)",
   "space": "O(n)",
   "explain": "Paths into a cell = paths from above + paths from the left; obstacle cells have 0.",
   "frames": [
    {
     "title": "Grid",
     "description": "1 marks an obstacle in the center.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "0"
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
     "title": "Row 0",
     "description": "One way to reach each cell: 1, 1, 1.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "paths",
       "cells": [
        [
         "1",
         "1",
         "1"
        ],
        [
         "·",
         "·",
         "·"
        ],
        [
         "·",
         "·",
         "·"
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
     "description": "1, then the obstacle → 0, then 0 + 1 = 1.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "paths",
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
         "·",
         "·",
         "·"
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
     "title": "Row 2",
     "description": "1; 0 + 1 = 1; 1 + 1 = 2.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "paths",
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
         "2"
        ]
       ],
       "hot": [
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "2 paths.",
     "codeLines": [
      8
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
