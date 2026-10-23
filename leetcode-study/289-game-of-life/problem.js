window.PROBLEM = {
 "id": "289",
 "title": "Game of Life",
 "level": "Medium",
 "topic": "Matrix",
 "question": "Each cell is alive (1) or dead (0). All cells update at once: a live cell survives with 2–3 live neighbours; a dead cell comes alive with exactly 3; everything else dies or stays dead.",
 "why": "The 4×3 sample board becomes the next generation shown on the right.",
 "idea": "Compute the next board from the old one (so updates don't affect each other), then copy it back.",
 "input": "[[0,1,0],[0,0,1],[1,1,1],[0,0,0]]",
 "output": "[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]",
 "notes": [],
 "best": "Compute the next board from the old one (so updates don't affect each other), then copy it back.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def gameOfLife(self, board):\n        m, n = len(board), len(board[0])\n        nxt = [[0] * n for _ in range(m)]\n        for i in range(m):\n            for j in range(n):\n                live = sum(board[a][b] for a in range(max(0, i-1), min(m, i+2)) for b in range(max(0, j-1), min(n, j+2))) - board[i][j]\n                nxt[i][j] = 1 if live == 3 or (board[i][j] and live == 2) else 0\n        board[:] = nxt",
   "time": "O(m·n)",
   "space": "O(m·n)",
   "explain": "Compute the next board from the old one (so updates don't affect each other), then copy it back.",
   "frames": [
    {
     "title": "Board",
     "description": "1 = alive.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "0",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "1"
        ],
        [
         "1",
         "1",
         "1"
        ],
        [
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
     "title": "Count neighbours",
     "description": "Cell (1,0) is dead with 3 live neighbours → becomes alive.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "0",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "1"
        ],
        [
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "1,0",
        "0,1",
        "1,2",
        "2,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Cell (0,1)",
     "description": "Alive with only 1 live neighbour → dies.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "0",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "1"
        ],
        [
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "0,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Next generation",
     "description": "Every cell updated using the OLD board.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "0",
         "0",
         "0"
        ],
        [
         "1",
         "0",
         "1"
        ],
        [
         "0",
         "1",
         "1"
        ],
        [
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
    }
   ]
  }
 ]
};
