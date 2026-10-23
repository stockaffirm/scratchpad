window.PROBLEM = {
 "id": "374",
 "title": "Guess Number Higher or Lower",
 "level": "Easy",
 "topic": "Binary Search",
 "question": "Guess the number I picked from 1…n using an API that says higher or lower.",
 "input": "n = 10, picked = 6",
 "output": "6",
 "notes": [
  "guess(mid) = -1 means the pick is lower than mid.",
  "Each call halves the search space."
 ],
 "best": "Classic binary search on the range 1…n.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def guessNumber(self, n):\n        lo, hi = 1, n\n        while lo <= hi:\n            mid = (lo + hi) // 2\n            g = guess(mid)\n            if g == 0: return mid\n            if g < 0: hi = mid - 1\n            else: lo = mid + 1",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Halve the range each guess.",
   "frames": [
    {
     "title": "Range",
     "description": "Pick is hidden in 1…10.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10"
       ],
       "hot": [],
       "marks": {
        "0": "lo",
        "9": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 5",
     "description": "Too low → lo = 6",
     "codeLines": [
      5,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10"
       ],
       "hot": [
        4
       ],
       "marks": {
        "0": "lo",
        "4": "mid",
        "9": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 8",
     "description": "Too high → hi = 7",
     "codeLines": [
      5,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10"
       ],
       "hot": [
        7
       ],
       "marks": {
        "5": "lo",
        "7": "mid",
        "9": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 6",
     "description": "Correct!",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "lo,mid",
        "6": "hi"
       }
      }
     ]
    }
   ]
  }
 ]
};
