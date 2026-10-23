window.PROBLEM = {
 "id": "14",
 "title": "Longest Common Prefix",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Find the longest beginning shared by all the words.",
 "why": "flower, flow, flight all start with 'fl'.",
 "idea": "Start with the first word as the prefix and shorten it until each other word starts with it.",
 "input": "[\"flower\",\"flow\",\"flight\"]",
 "output": "\"fl\"",
 "notes": [],
 "best": "Start with the first word as the prefix and shorten it until each other word starts with it.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestCommonPrefix(self, strs):\n        prefix = strs[0]\n        for s in strs[1:]:\n            while not s.startswith(prefix):\n                prefix = prefix[:-1]\n        return prefix",
   "time": "O(total chars)",
   "space": "O(1)",
   "explain": "Start with the first word as the prefix and shorten it until each other word starts with it.",
   "frames": [
    {
     "title": "Start",
     "description": "prefix = 'flower'.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "prefix",
       "cells": [
        "f",
        "l",
        "o",
        "w",
        "e",
        "r"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Check 'flow'",
     "description": "'flow' doesn't start with 'flower' → trim to 'flowe', 'flow'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "prefix",
       "cells": [
        "f",
        "l",
        "o",
        "w"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Check 'flight'",
     "description": "'flight' doesn't start with 'flow', 'flo' → trim to 'fl'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "prefix",
       "cells": [
        "f",
        "l"
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
     "title": "Return",
     "description": "'fl'",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "f",
        "l"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
