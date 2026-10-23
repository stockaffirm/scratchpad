window.PROBLEM = {
 "id": "69",
 "title": "Sqrt(x)",
 "level": "Easy",
 "topic": "Math",
 "question": "Compute the square root of x, rounded down to a whole number (no built-in sqrt).",
 "why": "sqrt(8) = 2.82… → 2.",
 "idea": "Binary search for the biggest number whose square is ≤ x.",
 "input": "8",
 "output": "2",
 "notes": [],
 "best": "Binary search for the biggest number whose square is ≤ x.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def mySqrt(self, x):\n        lo, hi = 0, x\n        while lo <= hi:\n            mid = (lo + hi) // 2\n            if mid * mid <= x: lo = mid + 1\n            else: hi = mid - 1\n        return hi",
   "time": "O(log x)",
   "space": "O(1)",
   "explain": "Binary search for the biggest number whose square is ≤ x.",
   "frames": [
    {
     "title": "Range",
     "description": "Answer is between 0 and 8.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "candidates",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8"
       ],
       "hot": [],
       "marks": {
        "0": "lo",
        "8": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 4",
     "description": "16 > 8 → too big → hi = 3.",
     "codeLines": [
      7
     ],
     "facts": {
      "lo": 0,
      "hi": 3
     },
     "rows": [
      {
       "name": "candidates",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "mid = 1",
     "description": "1 ≤ 8 → fine → lo = 2.",
     "codeLines": [
      6
     ],
     "facts": {
      "lo": 2,
      "hi": 3
     },
     "rows": [
      {
       "name": "candidates",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "mid = 2",
     "description": "4 ≤ 8 → lo = 3.",
     "codeLines": [
      6
     ],
     "facts": {
      "lo": 3,
      "hi": 3
     },
     "rows": [
      {
       "name": "candidates",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "mid = 3",
     "description": "9 > 8 → hi = 2. Now lo > hi.",
     "codeLines": [
      7
     ],
     "facts": {
      "lo": 3,
      "hi": 2
     },
     "rows": [
      {
       "name": "candidates",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8"
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
     "description": "hi = 2",
     "codeLines": [
      8
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
