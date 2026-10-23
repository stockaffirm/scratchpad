window.PROBLEM = {
 "id": "9",
 "title": "Palindrome Number",
 "level": "Easy",
 "topic": "Math",
 "question": "Is the integer the same when its digits are reversed?",
 "why": "121 reads the same both ways → True. −121 isn't (the minus sign) → False.",
 "idea": "Reverse only the second half of the digits and compare it with the first half.",
 "input": "121",
 "output": "True",
 "notes": [],
 "best": "Reverse only the second half of the digits and compare it with the first half.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isPalindrome(self, x):\n        if x < 0 or (x % 10 == 0 and x != 0): return False\n        rev = 0\n        while x > rev:\n            rev = rev * 10 + x % 10; x //= 10\n        return x == rev or x == rev // 10",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Reverse only the second half of the digits and compare it with the first half.",
   "frames": [
    {
     "title": "Start",
     "description": "x = 121, rev = 0.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "x",
       "cells": [
        "121"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "rev",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Move a digit",
     "description": "Take the last digit 1 → rev = 1, x = 12.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "x",
       "cells": [
        "12"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "rev",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Move a digit",
     "description": "Take 2 → rev = 12, x = 1. Now x ≤ rev → stop.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "x",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "rev",
       "cells": [
        "12"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Compare",
     "description": "x = 1 equals rev // 10 = 1 (middle digit ignored) → True.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
