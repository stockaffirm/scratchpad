window.PROBLEM = {
 "id": "433",
 "title": "Minimum Genetic Mutation",
 "level": "Medium",
 "topic": "Graph - BFS",
 "question": "A gene is an 8-letter string of A, C, G, T. A mutation changes ONE letter and the result must be in the bank. Fewest mutations from start to end?",
 "why": "AAAAACCC → AAAACCCC → AAACCCCC → AACCCCCC takes 3 mutations.",
 "idea": "BFS: try every single-letter change; keep the ones that are in the bank and unseen.",
 "input": "start=\"AAAAACCC\", end=\"AACCCCCC\", bank=[AAAACCCC, AAACCCCC, AACCCCCC]",
 "output": "3",
 "notes": [],
 "best": "BFS: try every single-letter change; keep the ones that are in the bank and unseen.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minMutation(self, start, end, bank):\n        bank, q, seen = set(bank), deque([(start, 0)]), {start}\n        while q:\n            g, d = q.popleft()\n            if g == end: return d\n            for i in range(len(g)):\n                for ch in 'ACGT':\n                    nxt = g[:i] + ch + g[i+1:]\n                    if nxt in bank and nxt not in seen:\n                        seen.add(nxt); q.append((nxt, d + 1))\n        return -1",
   "time": "O(8·4·bank)",
   "space": "O(bank)",
   "explain": "BFS: try every single-letter change; keep the ones that are in the bank and unseen.",
   "frames": [
    {
     "title": "Start",
     "description": "AAAAACCC; the bank has 3 valid genes.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "gene",
       "cells": [
        "AAAAACCC"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "1 mutation",
     "description": "Change the 5th letter → AAAACCCC (in the bank).",
     "codeLines": [
      9,
      10
     ],
     "facts": {
      "d": 1
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "AAAAACCC",
        "AAAACCCC"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "2 mutations",
     "description": "→ AAACCCCC",
     "codeLines": [
      11
     ],
     "facts": {
      "d": 2
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "AAAAACCC",
        "AAAACCCC",
        "AAACCCCC"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "3 mutations",
     "description": "→ AACCCCCC = end.",
     "codeLines": [
      6
     ],
     "facts": {
      "d": 3
     },
     "rows": [
      {
       "name": "path",
       "cells": [
        "AAAAACCC",
        "AAAACCCC",
        "AAACCCCC",
        "AACCCCCC"
       ],
       "hot": [
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "3",
     "codeLines": [
      6
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
