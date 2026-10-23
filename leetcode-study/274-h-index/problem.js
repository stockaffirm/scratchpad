window.PROBLEM = {
 "id": "274",
 "title": "H-Index",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Citations per paper. The h-index is the biggest h where h papers have at least h citations each.",
 "why": "[3,0,6,1,5] sorted high→low: 6,5,3,1,0. Three papers have ≥ 3 citations → h = 3.",
 "idea": "Sort descending, then count while the citation at position h is bigger than h.",
 "input": "[3,0,6,1,5]",
 "output": "3",
 "notes": [],
 "best": "Sort descending, then count while the citation at position h is bigger than h.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def hIndex(self, citations):\n        citations.sort(reverse=True)\n        h = 0\n        while h < len(citations) and citations[h] > h:\n            h += 1\n        return h",
   "time": "O(n log n)",
   "space": "O(1)",
   "explain": "Sort descending, then count while the citation at position h is bigger than h.",
   "frames": [
    {
     "title": "Sort",
     "description": "Highest citations first.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "citations",
       "cells": [
        "6",
        "5",
        "3",
        "1",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "h → 1",
     "description": "6 > 0 → at least 1 papers have ≥ 1 citations.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "h": 1
     },
     "rows": [
      {
       "name": "citations",
       "cells": [
        "6",
        "5",
        "3",
        "1",
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "h → 2",
     "description": "5 > 1 → at least 2 papers have ≥ 2 citations.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "h": 2
     },
     "rows": [
      {
       "name": "citations",
       "cells": [
        "6",
        "5",
        "3",
        "1",
        "0"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "h → 3",
     "description": "3 > 2 → at least 3 papers have ≥ 3 citations.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "h": 3
     },
     "rows": [
      {
       "name": "citations",
       "cells": [
        "6",
        "5",
        "3",
        "1",
        "0"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Stop",
     "description": "1 > 3 is false.",
     "codeLines": [
      5
     ],
     "facts": {
      "h": 3
     },
     "rows": [
      {
       "name": "citations",
       "cells": [
        "6",
        "5",
        "3",
        "1",
        "0"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "h-index = 3.",
     "codeLines": [
      7
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
