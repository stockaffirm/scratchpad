window.PROBLEM = {
 "id": "162",
 "title": "Find Peak Element",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "Find any index whose value is greater than both neighbours, in O(log n).",
 "input": "[1, 2, 1, 3, 5, 6, 4]",
 "output": "5 (index of 6)",
 "notes": [
  "If nums[mid] < nums[mid+1], a peak lies to the right.",
  "Otherwise mid itself or the left holds one."
 ],
 "best": "Binary search on slope: a peak always exists in the rising direction.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findPeakElement(self, nums):\n        lo, hi = 0, len(nums) - 1\n        while lo < hi:\n            mid = (lo + hi) // 2\n            if nums[mid] < nums[mid + 1]: lo = mid + 1\n            else: hi = mid\n        return lo",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Follow the uphill side.",
   "frames": [
    {
     "title": "Range",
     "description": "Search whole array.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "1",
        "3",
        "5",
        "6",
        "4"
       ],
       "hot": [],
       "marks": {
        "0": "lo",
        "6": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 3",
     "description": "3 < 5 → rising, go right",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "1",
        "3",
        "5",
        "6",
        "4"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {
        "0": "lo",
        "3": "mid",
        "6": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 5",
     "description": "6 > 4 → falling, go left",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "1",
        "3",
        "5",
        "6",
        "4"
       ],
       "hot": [
        5,
        6
       ],
       "marks": {
        "4": "lo",
        "5": "mid",
        "6": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 4",
     "description": "5 < 6 → rising, go right",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "1",
        "3",
        "5",
        "6",
        "4"
       ],
       "hot": [
        4,
        5
       ],
       "marks": {
        "4": "lo,mid",
        "5": "hi"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "lo = hi = 5 → value 6.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "1",
        "3",
        "5",
        "6",
        "4"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "peak"
       }
      }
     ]
    }
   ]
  }
 ]
};
