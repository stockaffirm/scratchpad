window.PROBLEM = {
 "id": "70",
 "title": "Climbing Stairs",
 "level": "Easy",
 "topic": "1D DP",
 "question": "You climb a staircase taking 1 or 2 steps at a time. How many different ways to reach step n?",
 "why": "n = 5 → 8 ways.",
 "idea": "Ways(n) = Ways(n−1) + Ways(n−2) — it's the Fibonacci sequence.",
 "input": "n = 5",
 "output": "8",
 "notes": [],
 "best": "Ways(n) = Ways(n−1) + Ways(n−2) — it's the Fibonacci sequence.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def climbStairs(self, n):\n        a, b = 1, 1\n        for _ in range(n - 1): a, b = b, a + b\n        return b",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Ways(n) = Ways(n−1) + Ways(n−2) — it's the Fibonacci sequence.",
   "frames": [
    {
     "title": "Start",
     "description": "1 way to be on step 0, and 1 way to reach step 1.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "ways",
       "cells": [
        "1",
        "1"
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
     "title": "Step 2",
     "description": "1 + 1 = 2",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "ways",
       "cells": [
        "1",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Step 3",
     "description": "1 + 2 = 3",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "ways",
       "cells": [
        "1",
        "1",
        "2",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Step 4",
     "description": "2 + 3 = 5",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "ways",
       "cells": [
        "1",
        "1",
        "2",
        "3",
        "5"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Step 5",
     "description": "3 + 5 = 8",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "ways",
       "cells": [
        "1",
        "1",
        "2",
        "3",
        "5",
        "8"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "8",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "8"
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
