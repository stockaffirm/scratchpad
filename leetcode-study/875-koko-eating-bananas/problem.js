window.PROBLEM = {
 "id": "875",
 "title": "Koko Eating Bananas",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "Smallest eating speed k (bananas/hour) to finish all piles within h hours.",
 "input": "piles=[3,6,7,11], h=8",
 "output": "4",
 "notes": [
  "hours(k) = Σ ceil(pile / k).",
  "Feasibility is monotonic in k."
 ],
 "best": "Binary search on the answer: faster speed always works if a slower one does.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minEatingSpeed(self, piles, h):\n        lo, hi = 1, max(piles)\n        while lo < hi:\n            mid = (lo + hi) // 2\n            if sum(ceil(p / mid) for p in piles) <= h: hi = mid\n            else: lo = mid + 1\n        return lo",
   "time": "O(n log max)",
   "space": "O(1)",
   "explain": "Binary search the answer (speed), not an index.",
   "frames": [
    {
     "title": "Speeds",
     "description": "Speed is between 1 and the biggest pile (11).",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "speed",
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
        "10",
        "11"
       ],
       "hot": [],
       "marks": {
        "0": "lo",
        "10": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 6",
     "description": "Hours needed = 6 ≤ 8 → try slower",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "hours": 6
     },
     "rows": [
      {
       "name": "speed",
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
        "10",
        "11"
       ],
       "hot": [
        5
       ],
       "marks": {
        "0": "lo",
        "5": "mid",
        "10": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 3",
     "description": "Hours needed = 10 > 8 → need faster",
     "codeLines": [
      5,
      7
     ],
     "facts": {
      "hours": 10
     },
     "rows": [
      {
       "name": "speed",
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
        "10",
        "11"
       ],
       "hot": [
        2
       ],
       "marks": {
        "0": "lo",
        "2": "mid",
        "5": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 5",
     "description": "Hours needed = 8 ≤ 8 → try slower",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "hours": 8
     },
     "rows": [
      {
       "name": "speed",
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
        "10",
        "11"
       ],
       "hot": [
        4
       ],
       "marks": {
        "3": "lo",
        "4": "mid",
        "5": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 4",
     "description": "Hours needed = 8 ≤ 8 → try slower",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "hours": 8
     },
     "rows": [
      {
       "name": "speed",
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
        "10",
        "11"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "lo,mid",
        "4": "hi"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Minimum speed = 4.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "speed",
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
        "10",
        "11"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "answer"
       }
      }
     ]
    }
   ]
  }
 ]
};
