window.PROBLEM = {
 "id": "1137",
 "title": "N-th Tribonacci Number",
 "level": "Easy",
 "topic": "DP - 1D",
 "question": "T0=0, T1=1, T2=1, Tn = Tn-1 + Tn-2 + Tn-3. Find Tn.",
 "input": "n = 4",
 "output": "4",
 "notes": [
  "Only the previous three terms matter.",
  "n = 1 and n = 2 both return 1."
 ],
 "best": "Rolling three variables instead of an array.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def tribonacci(self, n):\n        if n == 0: return 0\n        a, b, c = 0, 1, 1\n        for _ in range(n - 2):\n            a, b, c = b, c, a + b + c\n        return c",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Keep only the last three values.",
   "frames": [
    {
     "title": "Base",
     "description": "T0=0, T1=1, T2=1.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "T",
       "cells": [
        "0",
        "1",
        "1"
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
     "title": "Step 1",
     "description": "T3 = 0 + 1 + 1 = 2.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "T",
       "cells": [
        "0",
        "1",
        "1",
        "2"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Step 2",
     "description": "T4 = 1 + 1 + 2 = 4.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "T",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "4"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "T4 = 4",
     "codeLines": [
      7
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
