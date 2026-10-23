window.PROBLEM = {
 "id": "201",
 "title": "Bitwise AND of Numbers Range",
 "level": "Medium",
 "topic": "Bit Manipulation",
 "question": "What is the bitwise AND of every number from left to right?",
 "why": "5 AND 6 AND 7 = 4.",
 "idea": "The result is the common binary prefix of left and right. Shift both right until they match.",
 "input": "left=5 (101), right=7 (111)",
 "output": "4",
 "notes": [],
 "best": "The result is the common binary prefix of left and right. Shift both right until they match.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def rangeBitwiseAnd(self, left, right):\n        shift = 0\n        while left != right:\n            left >>= 1; right >>= 1; shift += 1\n        return left << shift",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "The result is the common binary prefix of left and right. Shift both right until they match.",
   "frames": [
    {
     "title": "Start",
     "description": "left = 101, right = 111.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "left",
       "cells": [
        "1",
        "0",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "right",
       "cells": [
        "1",
        "1",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Shift 1",
     "description": "10 vs 11 — still different.",
     "codeLines": [
      5
     ],
     "facts": {
      "shift": 1
     },
     "rows": [
      {
       "name": "left",
       "cells": [
        "1",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "right",
       "cells": [
        "1",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Shift 2",
     "description": "1 vs 1 — equal! Common prefix = 1.",
     "codeLines": [
      4
     ],
     "facts": {
      "shift": 2
     },
     "rows": [
      {
       "name": "left",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "right",
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
     "title": "Return",
     "description": "1 << 2 = 100 = 4.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "0",
        "0"
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
