window.PROBLEM = {
 "id": "79",
 "title": "Word Search",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "Can you spell the word by moving through neighbouring cells (up/down/left/right) without reusing a cell?",
 "why": "On the sample grid, ABCCED can be traced: A→B→C→C→E→D.",
 "idea": "DFS from every cell; mark the cell used while exploring, unmark when backing out.",
 "input": "board 3×4, word=\"ABCCED\"",
 "output": "True",
 "notes": [],
 "best": "DFS from every cell; mark the cell used while exploring, unmark when backing out.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def exist(self, board, word):\n        m, n = len(board), len(board[0])\n        def dfs(r, c, i):\n            if i == len(word): return True\n            if not (0 <= r < m and 0 <= c < n) or board[r][c] != word[i]: return False\n            tmp, board[r][c] = board[r][c], '#'\n            found = any(dfs(r + dr, c + dc, i + 1) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)))\n            board[r][c] = tmp\n            return found\n        return any(dfs(r, c, 0) for r in range(m) for c in range(n))",
   "time": "O(m·n·3^L)",
   "space": "O(L)",
   "explain": "DFS from every cell; mark the cell used while exploring, unmark when backing out.",
   "frames": [
    {
     "title": "Board",
     "description": "Find A-B-C-C-E-D.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "A",
         "B",
         "C",
         "E"
        ],
        [
         "S",
         "F",
         "C",
         "S"
        ],
        [
         "A",
         "D",
         "E",
         "E"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "A → B → C",
     "description": "(0,0), (0,1), (0,2).",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "A",
         "B",
         "C",
         "E"
        ],
        [
         "S",
         "F",
         "C",
         "S"
        ],
        [
         "A",
         "D",
         "E",
         "E"
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
     "title": "→ C → E",
     "description": "Down to (1,2) = C, then (2,2) = E.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "A",
         "B",
         "C",
         "E"
        ],
        [
         "S",
         "F",
         "C",
         "S"
        ],
        [
         "A",
         "D",
         "E",
         "E"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "0,2",
        "1,2",
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "→ D",
     "description": "Left to (2,1) = D. The whole word is spelled.",
     "codeLines": [
      5
     ],
     "facts": {
      "result": "True"
     },
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "A",
         "B",
         "C",
         "E"
        ],
        [
         "S",
         "F",
         "C",
         "S"
        ],
        [
         "A",
         "D",
         "E",
         "E"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "0,2",
        "1,2",
        "2,2",
        "2,1"
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
