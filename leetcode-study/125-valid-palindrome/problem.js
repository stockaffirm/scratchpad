window.PROBLEM = {
 "id": "125",
 "title": "Valid Palindrome",
 "level": "Easy",
 "topic": "Two Pointers",
 "question": "Ignoring spaces, punctuation and capital letters, does the text read the same forwards and backwards?",
 "why": "'A man, a plan, a canal: Panama' becomes amanaplanacanalpanama, which is a palindrome → True.",
 "idea": "Two pointers from the ends moving inward; every pair must match.",
 "input": "\"A man, a plan, a canal: Panama\"",
 "output": "True",
 "notes": [],
 "best": "Two pointers from the ends moving inward; every pair must match.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isPalindrome(self, s):\n        t = [c.lower() for c in s if c.isalnum()]\n        l, r = 0, len(t) - 1\n        while l < r:\n            if t[l] != t[r]: return False\n            l += 1; r -= 1\n        return True",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Two pointers from the ends moving inward; every pair must match.",
   "frames": [
    {
     "title": "Clean",
     "description": "Keep only letters/digits, lower-case.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "t",
       "cells": [
        "a",
        "m",
        "a",
        "n",
        "a",
        "p",
        "l",
        "a",
        "n",
        "a",
        "c",
        "a",
        "n",
        "a",
        "l",
        "p",
        "a",
        "n",
        "a",
        "m",
        "a"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Compare ends",
     "description": "'a' = 'a' ✓",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "t",
       "cells": [
        "a",
        "m",
        "a",
        "n",
        "a",
        "p",
        "l",
        "a",
        "n",
        "a",
        "c",
        "a",
        "n",
        "a",
        "l",
        "p",
        "a",
        "n",
        "a",
        "m",
        "a"
       ],
       "hot": [
        0,
        20
       ],
       "marks": {
        "0": "l",
        "20": "r"
       }
      }
     ]
    },
    {
     "title": "Compare ends",
     "description": "'m' = 'm' ✓",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "t",
       "cells": [
        "a",
        "m",
        "a",
        "n",
        "a",
        "p",
        "l",
        "a",
        "n",
        "a",
        "c",
        "a",
        "n",
        "a",
        "l",
        "p",
        "a",
        "n",
        "a",
        "m",
        "a"
       ],
       "hot": [
        1,
        19
       ],
       "marks": {
        "1": "l",
        "19": "r"
       }
      }
     ]
    },
    {
     "title": "Compare ends",
     "description": "'a' = 'a' ✓",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "t",
       "cells": [
        "a",
        "m",
        "a",
        "n",
        "a",
        "p",
        "l",
        "a",
        "n",
        "a",
        "c",
        "a",
        "n",
        "a",
        "l",
        "p",
        "a",
        "n",
        "a",
        "m",
        "a"
       ],
       "hot": [
        2,
        18
       ],
       "marks": {
        "2": "l",
        "18": "r"
       }
      }
     ]
    },
    {
     "title": "Keep going",
     "description": "Every remaining pair matches until the pointers meet.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "t",
       "cells": [
        "a",
        "m",
        "a",
        "n",
        "a",
        "p",
        "l",
        "a",
        "n",
        "a",
        "c",
        "a",
        "n",
        "a",
        "l",
        "p",
        "a",
        "n",
        "a",
        "m",
        "a"
       ],
       "hot": [
        10
       ],
       "marks": {
        "10": "l,r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Palindrome.",
     "codeLines": [
      8
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
