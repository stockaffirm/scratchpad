window.PROBLEM = {
 "id": "427",
 "title": "Construct Quad Tree",
 "level": "Medium",
 "topic": "Divide & Conquer",
 "question": "Compress a square grid of 0s and 1s into a quad tree: if a block is all the same value it's a leaf; otherwise split it into 4 quadrants.",
 "why": "The sample 4×4 grid isn't uniform, so it splits into four 2×2 quadrants — each of which is uniform (1, 0, 1, 1).",
 "idea": "Check if the block is uniform. If not, split into 4 and recurse.",
 "input": "[[1,1,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]]",
 "output": "root with 4 leaf children: 1, 0, 1, 1",
 "notes": [],
 "best": "Check if the block is uniform. If not, split into 4 and recurse.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def construct(self, grid):\n        def build(r, c, size):\n            v = grid[r][c]\n            if all(grid[i][j] == v for i in range(r, r + size) for j in range(c, c + size)):\n                return Node(v == 1, True, None, None, None, None)\n            h = size // 2\n            return Node(True, False, build(r, c, h), build(r, c + h, h), build(r + h, c, h), build(r + h, c + h, h))\n        return build(0, 0, len(grid))",
   "time": "O(n² log n)",
   "space": "O(log n)",
   "explain": "Check if the block is uniform. If not, split into 4 and recurse.",
   "frames": [
    {
     "title": "Grid",
     "description": "Is the whole 4×4 block one value?",
     "codeLines": [
      3
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
         "0"
        ],
        [
         "1",
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "1",
         "1",
         "1"
        ],
        [
         "1",
         "1",
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
     "title": "Not uniform",
     "description": "Mixed 1s and 0s → split into 4 quadrants.",
     "codeLines": [
      7
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
         "0"
        ],
        [
         "1",
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "1",
         "1",
         "1"
        ],
        [
         "1",
         "1",
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
     "title": "Quadrants",
     "description": "Top-left all 1, top-right all 0, bottom-left all 1, bottom-right all 1 → each is a leaf.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "children",
       "cells": [
        "1",
        "0",
        "1",
        "1"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Root (not a leaf) with four leaf children.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "root",
       "cells": [
        "split"
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
