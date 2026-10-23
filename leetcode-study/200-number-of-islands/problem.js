window.PROBLEM = {
 "id": "200",
 "title": "Number of Islands",
 "level": "Medium",
 "topic": "Graph - General",
 "question": "A grid of land (1) and water (0). An island is land connected up/down/left/right. How many islands?",
 "why": "The sample grid has three separate groups of 1s → 3.",
 "idea": "Scan the grid; when you hit unvisited land, count an island and 'sink' all connected land with DFS.",
 "input": "[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]",
 "output": "3",
 "notes": [],
 "best": "Scan the grid; when you hit unvisited land, count an island and 'sink' all connected land with DFS.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def numIslands(self, grid):\n        def sink(r, c):\n            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] != '1': return\n            grid[r][c] = '0'\n            sink(r+1, c); sink(r-1, c); sink(r, c+1); sink(r, c-1)\n        count = 0\n        for r in range(len(grid)):\n            for c in range(len(grid[0])):\n                if grid[r][c] == '1': sink(r, c); count += 1\n        return count",
   "time": "O(m·n)",
   "space": "O(m·n)",
   "explain": "Scan the grid; when you hit unvisited land, count an island and 'sink' all connected land with DFS.",
   "frames": [
    {
     "title": "Grid",
     "description": "1 = land, 0 = water.",
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
         "1",
         "0",
         "0",
         "0"
        ],
        [
         "1",
         "1",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "1",
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
     "title": "Island 1",
     "description": "Found land at (0,0): sink the whole 2×2 block (mark A).",
     "codeLines": [
      10,
      6
     ],
     "facts": {
      "count": 1
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "A",
         "A",
         "0",
         "0",
         "0"
        ],
        [
         "A",
         "A",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "1",
         "1"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "1,0",
        "1,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Island 2",
     "description": "Land at (2,2): sink it (B).",
     "codeLines": [
      10
     ],
     "facts": {
      "count": 2
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "A",
         "A",
         "0",
         "0",
         "0"
        ],
        [
         "A",
         "A",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "B",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "1",
         "1"
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
     "title": "Island 3",
     "description": "Land at (3,3) and (3,4): sink both (C).",
     "codeLines": [
      10
     ],
     "facts": {
      "count": 3
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "A",
         "A",
         "0",
         "0",
         "0"
        ],
        [
         "A",
         "A",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "B",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "C",
         "C"
        ]
       ],
       "hot": [
        "3,3",
        "3,4"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "3 islands.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "3"
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
