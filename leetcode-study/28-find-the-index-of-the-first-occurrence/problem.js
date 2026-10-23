window.PROBLEM = {
 "id": "28",
 "title": "Find the Index of the First Occurrence in a String",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Find where the word needle first appears inside haystack. Return −1 if it doesn't.",
 "why": "haystack 'hello', needle 'll' → starts at index 2.",
 "idea": "Slide a window the size of the needle along the haystack and compare.",
 "input": "haystack=\"hello\", needle=\"ll\"",
 "output": "2",
 "notes": [],
 "best": "Slide a window the size of the needle along the haystack and compare.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def strStr(self, haystack, needle):\n        for i in range(len(haystack) - len(needle) + 1):\n            if haystack[i:i+len(needle)] == needle:\n                return i\n        return -1",
   "time": "O(n·m)",
   "space": "O(1)",
   "explain": "Slide a window the size of the needle along the haystack and compare.",
   "frames": [
    {
     "title": "Start",
     "description": "Compare each 2-letter window with 'll'.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "haystack",
       "cells": [
        "h",
        "e",
        "l",
        "l",
        "o"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 0",
     "description": "'he' ≠ 'll'.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "haystack",
       "cells": [
        "h",
        "e",
        "l",
        "l",
        "o"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "0": "i"
       }
      }
     ]
    },
    {
     "title": "i = 1",
     "description": "'el' ≠ 'll'.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "haystack",
       "cells": [
        "h",
        "e",
        "l",
        "l",
        "o"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {
        "1": "i"
       }
      }
     ]
    },
    {
     "title": "i = 2",
     "description": "'ll' == 'll' → found!",
     "codeLines": [
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "haystack",
       "cells": [
        "h",
        "e",
        "l",
        "l",
        "o"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {
        "2": "i"
       }
      }
     ]
    }
   ]
  }
 ]
};
