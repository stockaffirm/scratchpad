window.PROBLEM = {
 "id": "212",
 "title": "Word Search II",
 "level": "Hard",
 "topic": "Trie",
 "question": "Find all words from a list that can be traced on a letter grid by moving up/down/left/right (no cell used twice in a word).",
 "why": "On the sample board, 'oath' and 'eat' can be traced; 'pea' and 'rain' can't.",
 "idea": "Put the words in a trie, then DFS from every cell, following the trie so dead-end paths stop early.",
 "input": "board 4×4, words=[\"oath\",\"pea\",\"eat\",\"rain\"]",
 "output": "[\"oath\",\"eat\"]",
 "notes": [],
 "best": "Put the words in a trie, then DFS from every cell, following the trie so dead-end paths stop early.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findWords(self, board, words):\n        trie = {}\n        for w in words:\n            node = trie\n            for ch in w: node = node.setdefault(ch, {})\n            node['$'] = w\n        res, m, n = [], len(board), len(board[0])\n        def dfs(r, c, node):\n            ch = board[r][c]\n            if ch not in node: return\n            nxt = node[ch]\n            if '$' in nxt: res.append(nxt.pop('$'))\n            board[r][c] = '#'\n            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n                nr, nc = r + dr, c + dc\n                if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != '#': dfs(nr, nc, nxt)\n            board[r][c] = ch\n        for r in range(m):\n            for c in range(n): dfs(r, c, trie)\n        return res",
   "time": "O(m·n·4^L)",
   "space": "O(total letters)",
   "explain": "Put the words in a trie, then DFS from every cell, following the trie so dead-end paths stop early.",
   "frames": [
    {
     "title": "Board",
     "description": "Words: oath, pea, eat, rain (stored in a trie).",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "o",
         "a",
         "a",
         "n"
        ],
        [
         "e",
         "t",
         "a",
         "e"
        ],
        [
         "i",
         "h",
         "k",
         "r"
        ],
        [
         "i",
         "f",
         "l",
         "v"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Trace 'oath'",
     "description": "o (0,0) → a (0,1) → t (1,1) → h (2,1): a full word!",
     "codeLines": [
      13
     ],
     "facts": {
      "found": "oath"
     },
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "o",
         "a",
         "a",
         "n"
        ],
        [
         "e",
         "t",
         "a",
         "e"
        ],
        [
         "i",
         "h",
         "k",
         "r"
        ],
        [
         "i",
         "f",
         "l",
         "v"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "1,1",
        "2,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Trace 'eat'",
     "description": "e (1,3) → a (1,2) → t (1,1): another word!",
     "codeLines": [
      13
     ],
     "facts": {
      "found": "oath, eat"
     },
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "o",
         "a",
         "a",
         "n"
        ],
        [
         "e",
         "t",
         "a",
         "e"
        ],
        [
         "i",
         "h",
         "k",
         "r"
        ],
        [
         "i",
         "f",
         "l",
         "v"
        ]
       ],
       "hot": [
        "1,3",
        "1,2",
        "1,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Dead ends",
     "description": "No path spells 'pea' or 'rain' — the trie stops those searches early.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "oath",
        "eat"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
