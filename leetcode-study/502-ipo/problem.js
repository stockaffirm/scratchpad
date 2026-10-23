window.PROBLEM = {
 "id": "502",
 "title": "IPO",
 "level": "Hard",
 "topic": "Heap",
 "question": "You start with capital w and can do at most k projects. Each project needs some minimum capital and gives a profit (added to your capital). Maximise final capital.",
 "why": "k=2, w=0, profits [1,2,3], capital [0,1,1] → do project 0 (+1 → w=1), then project 2 (+3 → w=4).",
 "idea": "Each round, among projects you can afford, take the most profitable one (max-heap).",
 "input": "k=2, w=0, profits=[1,2,3], capital=[0,1,1]",
 "output": "4",
 "notes": [],
 "best": "Each round, among projects you can afford, take the most profitable one (max-heap).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findMaximizedCapital(self, k, w, profits, capital):\n        projects = sorted(zip(capital, profits)); heap = []; i = 0\n        for _ in range(k):\n            while i < len(projects) and projects[i][0] <= w:\n                heappush(heap, -projects[i][1]); i += 1\n            if not heap: break\n            w -= heappop(heap)\n        return w",
   "time": "O(n log n)",
   "space": "O(n)",
   "explain": "Each round, among projects you can afford, take the most profitable one (max-heap).",
   "frames": [
    {
     "title": "Sort by cost",
     "description": "(capital, profit): (0,1), (1,2), (1,3).",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "projects",
       "cells": [
        "(0,1)",
        "(1,2)",
        "(1,3)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "capital w",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Round 1",
     "description": "Only (0,1) is affordable → profit 1 → w = 1.",
     "codeLines": [
      5,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "affordable profits",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "capital w",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Round 2",
     "description": "Now (1,2) and (1,3) are affordable → take the best profit 3 → w = 4.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "affordable profits",
       "cells": [
        "3",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "capital w",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "k = 2 rounds done → 4.",
     "codeLines": [
      9
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
