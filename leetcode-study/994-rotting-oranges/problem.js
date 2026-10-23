window.PROBLEM = {
 "id": "994",
 "title": "Rotting Oranges",
 "level": "Medium",
 "topic": "Graphs - BFS",
 "question": "Each minute rotten oranges (2) rot adjacent fresh ones (1). Minutes until none are fresh?",
 "input": "[[2,1,1],[1,1,0],[0,1,1]]",
 "output": "4",
 "notes": [
  "Start BFS from every rotten orange at once.",
  "Leftover fresh → -1."
 ],
 "best": "Multi-source BFS: all rotten oranges spread together each minute.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def orangesRotting(self, grid):\n        q, fresh = deque(), 0\n        for r, row in enumerate(grid):\n            for c, v in enumerate(row):\n                if v == 2: q.append((r, c))\n                elif v == 1: fresh += 1\n        minutes = 0\n        while q and fresh:\n            for _ in range(len(q)):\n                r, c = q.popleft()\n                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n                    nr, nc = r + dr, c + dc\n                    if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 1:\n                        grid[nr][nc] = 2; fresh -= 1; q.append((nr, nc))\n            minutes += 1\n        return minutes if fresh == 0 else -1",
   "time": "O(m·n)",
   "space": "O(m·n)",
   "explain": "Multi-source BFS, one minute per level.",
   "frames": [
    {
     "title": "Start",
     "description": "One rotten orange; 6 fresh.",
     "codeLines": [
      3,
      7
     ],
     "facts": {
      "fresh": 6,
      "minutes": 0
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "2",
         "1",
         "1"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
         "0",
         "1",
         "1"
        ]
       ],
       "hot": [
        "0,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Minute 1",
     "description": "Rot spreads to 2 oranges.",
     "codeLines": [
      16,
      15
     ],
     "facts": {
      "fresh": 4,
      "minutes": 1
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "2",
         "2",
         "1"
        ],
        [
         "2",
         "1",
         "0"
        ],
        [
         "0",
         "1",
         "1"
        ]
       ],
       "hot": [
        "1,0",
        "0,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Minute 2",
     "description": "Rot spreads to 2 oranges.",
     "codeLines": [
      16,
      15
     ],
     "facts": {
      "fresh": 2,
      "minutes": 2
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "2",
         "2",
         "2"
        ],
        [
         "2",
         "2",
         "0"
        ],
        [
         "0",
         "1",
         "1"
        ]
       ],
       "hot": [
        "1,1",
        "0,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Minute 3",
     "description": "Rot spreads to 1 orange.",
     "codeLines": [
      16,
      15
     ],
     "facts": {
      "fresh": 1,
      "minutes": 3
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "2",
         "2",
         "2"
        ],
        [
         "2",
         "2",
         "0"
        ],
        [
         "0",
         "2",
         "1"
        ]
       ],
       "hot": [
        "2,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Minute 4",
     "description": "Rot spreads to 1 orange.",
     "codeLines": [
      16,
      15
     ],
     "facts": {
      "fresh": 0,
      "minutes": 4
     },
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "2",
         "2",
         "2"
        ],
        [
         "2",
         "2",
         "0"
        ],
        [
         "0",
         "2",
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
     "description": "No fresh left after 4 minutes.",
     "codeLines": [
      17
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
