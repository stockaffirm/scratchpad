window.PROBLEM = {
 "id": "17",
 "title": "Letter Combinations of a Phone Number",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "Return all letter combinations a digit string could represent on a phone keypad.",
 "input": "\"23\"",
 "output": "[\"ad\",\"ae\",\"af\",\"bd\",\"be\",\"bf\",\"cd\",\"ce\",\"cf\"]",
 "notes": [
  "Depth = number of digits.",
  "path + ch builds a new string, so no explicit undo is needed."
 ],
 "best": "Backtracking: one level per digit, one branch per letter.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def letterCombinations(self, digits):\n        if not digits: return []\n        phone = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}\n        res = []\n        def go(i, path):\n            if i == len(digits): res.append(path); return\n            for ch in phone[digits[i]]: go(i + 1, path + ch)\n        go(0, '')\n        return res",
   "time": "O(4ⁿ·n)",
   "space": "O(n)",
   "explain": "Choose one letter per digit, recurse, backtrack.",
   "frames": [
    {
     "title": "Digits",
     "description": "2 → abc, 3 → def",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "digits",
       "cells": [
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pick 'a'",
     "description": "Then try d, e, f for digit 3.",
     "codeLines": [
      8,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "a"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af"
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
     "title": "Pick 'b'",
     "description": "Then try d, e, f for digit 3.",
     "codeLines": [
      8,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "b"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf"
       ],
       "hot": [
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pick 'c'",
     "description": "Then try d, e, f for digit 3.",
     "codeLines": [
      8,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "c"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf",
        "cd",
        "ce",
        "cf"
       ],
       "hot": [
        6,
        7,
        8
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "9 combinations.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf",
        "cd",
        "ce",
        "cf"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
