window.PROBLEM = {
 "id": "909",
 "title": "Snakes and Ladders",
 "level": "Medium",
 "topic": "Graph - BFS",
 "question": "Play snakes and ladders on an n×n board (squares numbered in a zigzag). Each move you roll 1–6; ladders/snakes move you. Fewest moves to reach the last square?",
 "why": "3×3 board with a ladder from 2 to 8: roll 1 (→2 → ladder to 8), then roll 1 → 9. That's 2 moves.",
 "idea": "BFS over squares: each square connects to the next 6 squares; the first time you reach the end is the minimum.",
 "input": "3×3 board with a ladder 2 → 8",
 "output": "2",
 "notes": [],
 "best": "BFS over squares: each square connects to the next 6 squares; the first time you reach the end is the minimum.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def snakesAndLadders(self, board):\n        n = len(board)\n        def cell(s):\n            r, c = divmod(s - 1, n)\n            row = n - 1 - r\n            col = c if r % 2 == 0 else n - 1 - c\n            return row, col\n        seen, q = {1}, deque([(1, 0)])\n        while q:\n            s, d = q.popleft()\n            for nxt in range(s + 1, min(s + 6, n * n) + 1):\n                r, c = cell(nxt)\n                if board[r][c] != -1: nxt = board[r][c]\n                if nxt == n * n: return d + 1\n                if nxt not in seen: seen.add(nxt); q.append((nxt, d + 1))\n        return -1",
   "time": "O(n²)",
   "space": "O(n²)",
   "explain": "BFS over squares: each square connects to the next 6 squares; the first time you reach the end is the minimum.",
   "frames": [
    {
     "title": "Board",
     "description": "Squares 1–9 in a zigzag. Landing on 2 takes a ladder up to 8.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "board (top → bottom)",
       "cells": [
        [
         "7",
         "8",
         "9"
        ],
        [
         "6",
         "5",
         "4"
        ],
        [
         "1",
         "2→8",
         "3"
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
     "title": "Start",
     "description": "Square 1 after 0 moves.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "reachable in 0",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Moves = 1",
     "description": "Roll 1 → square 2 → ladder to 8. Rolls 2–6 → squares 3,4,5,6,7.",
     "codeLines": [
      14,
      16
     ],
     "facts": {},
     "rows": [
      {
       "name": "reachable in 1",
       "cells": [
        "8",
        "3",
        "4",
        "5",
        "6",
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Moves = 2",
     "description": "From 8 roll 1 → square 9 = the end.",
     "codeLines": [
      15
     ],
     "facts": {},
     "rows": [
      {
       "name": "reachable in 2",
       "cells": [
        "9"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Minimum 2 moves.",
     "codeLines": [
      15
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
