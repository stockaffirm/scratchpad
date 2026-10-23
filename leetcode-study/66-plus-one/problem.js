window.PROBLEM = {
 "id": "66",
 "title": "Plus One",
 "level": "Easy",
 "topic": "Math",
 "question": "A number is stored as a list of digits. Add one to it.",
 "why": "[1,2,9] means 129 → 130 → [1,3,0].",
 "idea": "From the right: a 9 becomes 0 and carries; the first non-9 digit gets +1.",
 "input": "[1,2,9]",
 "output": "[1,3,0]",
 "notes": [],
 "best": "From the right: a 9 becomes 0 and carries; the first non-9 digit gets +1.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def plusOne(self, digits):\n        for i in range(len(digits) - 1, -1, -1):\n            if digits[i] < 9:\n                digits[i] += 1; return digits\n            digits[i] = 0\n        return [1] + digits",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "From the right: a 9 becomes 0 and carries; the first non-9 digit gets +1.",
   "frames": [
    {
     "title": "Start",
     "description": "Look at the last digit.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "digits",
       "cells": [
        "1",
        "2",
        "9"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "9 → 0",
     "description": "9 can't grow → set to 0 and carry.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "digits",
       "cells": [
        "1",
        "2",
        "0"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "2 → 3",
     "description": "2 < 9 → add 1 and stop.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "digits",
       "cells": [
        "1",
        "3",
        "0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "[1, 3, 0]",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "3",
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
    }
   ]
  }
 ]
};
