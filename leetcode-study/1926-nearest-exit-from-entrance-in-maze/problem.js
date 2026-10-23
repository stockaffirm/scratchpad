window.PROBLEM = {
 "id": "1926",
 "title": "Nearest Exit from Entrance in Maze",
 "level": "Medium",
 "topic": "Graphs - BFS",
 "question": "Fewest steps from the entrance to any border exit ('.' cells on the edge).",
 "input": "4×4 maze, entrance (1,0)",
 "output": "4",
 "notes": [
  "The entrance itself doesn't count as an exit.",
  "Mark cells visited by turning them into walls."
 ],
 "best": "BFS: the first border cell reached is the shortest path.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def nearestExit(self, maze, entrance):\n        m, n = len(maze), len(maze[0])\n        q = deque([(entrance[0], entrance[1], 0)])\n        maze[entrance[0]][entrance[1]] = '+'\n        while q:\n            r, c, d = q.popleft()\n            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n                nr, nc = r + dr, c + dc\n                if 0 <= nr < m and 0 <= nc < n and maze[nr][nc] == '.':\n                    if nr in (0, m-1) or nc in (0, n-1): return d + 1\n                    maze[nr][nc] = '+'\n                    q.append((nr, nc, d + 1))\n        return -1",
   "time": "O(m·n)",
   "space": "O(m·n)",
   "explain": "BFS expands one step at a time, so the first exit found is nearest.",
   "frames": [
    {
     "title": "Maze",
     "description": "Start S at (1,0). '+' is a wall; the border '.' cell is the exit.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "maze",
       "cells": [
        [
         "+",
         "+",
         "+",
         "+"
        ],
        [
         "S",
         ".",
         ".",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
        ]
       ],
       "hot": [
        "1,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Step 1",
     "description": "Expand to (1,1). Not on the border.",
     "codeLines": [
      7,
      13
     ],
     "facts": {
      "d": 1
     },
     "rows": [
      {
       "name": "maze",
       "cells": [
        [
         "+",
         "+",
         "+",
         "+"
        ],
        [
         "S",
         "1",
         ".",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
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
     "title": "Step 2",
     "description": "Expand to (1,2).",
     "codeLines": [
      13
     ],
     "facts": {
      "d": 2
     },
     "rows": [
      {
       "name": "maze",
       "cells": [
        [
         "+",
         "+",
         "+",
         "+"
        ],
        [
         "S",
         "1",
         "2",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
        ]
       ],
       "hot": [
        "1,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Step 3",
     "description": "Expand down to (2,2).",
     "codeLines": [
      13
     ],
     "facts": {
      "d": 3
     },
     "rows": [
      {
       "name": "maze",
       "cells": [
        [
         "+",
         "+",
         "+",
         "+"
        ],
        [
         "S",
         "1",
         "2",
         "+"
        ],
        [
         "+",
         "+",
         "3",
         "+"
        ],
        [
         "+",
         "+",
         ".",
         "+"
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
     "title": "Exit found",
     "description": "(3,2) is on the border → d + 1 = 4.",
     "codeLines": [
      11
     ],
     "facts": {
      "answer": 4
     },
     "rows": [
      {
       "name": "maze",
       "cells": [
        [
         "+",
         "+",
         "+",
         "+"
        ],
        [
         "S",
         "1",
         "2",
         "+"
        ],
        [
         "+",
         "+",
         "3",
         "+"
        ],
        [
         "+",
         "+",
         "E",
         "+"
        ]
       ],
       "hot": [
        "3,2"
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
