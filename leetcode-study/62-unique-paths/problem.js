window.PROBLEM = {
 "id": "62",
 "title": "Unique Paths",
 "level": "Medium",
 "topic": "DP - Multidimensional",
 "question": "A robot moves only right or down from top-left to bottom-right. How many paths?",
 "input": "m = 3, n = 3",
 "output": "6",
 "notes": [
  "Top row and left column are all 1.",
  "row[j] (old) is 'from above'; row[j-1] is 'from left'."
 ],
 "best": "One rolling row: each cell adds the cell to its left.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def uniquePaths(self, m, n):\n        row = [1] * n\n        for _ in range(1, m):\n            for j in range(1, n):\n                row[j] += row[j - 1]\n        return row[-1]",
   "time": "O(m·n)",
   "space": "O(n)",
   "explain": "Paths to a cell = paths from above + paths from the left.",
   "frames": [
    {
     "title": "Edges",
     "description": "Only one way along the top row and left column.",
     "codeLines": [
      3
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
         "·",
         "·"
        ],
        [
         "1",
         "·",
         "·"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "0,2",
        "1,0",
        "2,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 2",
     "description": "2 = 1+1,  3 = 2+1.",
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
         "2",
         "3"
        ],
        [
         "1",
         "·",
         "·"
        ]
       ],
       "hot": [
        "1,1",
        "1,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 3",
     "description": "3 = 2+1,  6 = 3+3.",
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
         "2",
         "3"
        ],
        [
         "1",
         "3",
         "6"
        ]
       ],
       "hot": [
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
     "description": "6 paths.",
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
         "2",
         "3"
        ],
        [
         "1",
         "3",
         "6"
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
