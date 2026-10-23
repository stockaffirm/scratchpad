window.PROBLEM = {
 "id": "153",
 "title": "Find Minimum in Rotated Sorted Array",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "A sorted list was rotated. Find the smallest value in O(log n).",
 "why": "[3,4,5,1,2] → the minimum is 1.",
 "idea": "Compare the middle with the right end: if the middle is bigger, the minimum is to the right; otherwise it's at mid or to the left.",
 "input": "[3,4,5,1,2]",
 "output": "1",
 "notes": [],
 "best": "Compare the middle with the right end: if the middle is bigger, the minimum is to the right; otherwise it's at mid or to the left.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findMin(self, nums):\n        lo, hi = 0, len(nums) - 1\n        while lo < hi:\n            mid = (lo + hi) // 2\n            if nums[mid] > nums[hi]: lo = mid + 1\n            else: hi = mid\n        return nums[lo]",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Compare the middle with the right end: if the middle is bigger, the minimum is to the right; otherwise it's at mid or to the left.",
   "frames": [
    {
     "title": "Start",
     "description": "Whole list.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "4",
        "5",
        "1",
        "2"
       ],
       "hot": [],
       "marks": {
        "0": "lo",
        "4": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 2",
     "description": "5 > 2 (right end) → the drop is to the right → lo = 3.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "4",
        "5",
        "1",
        "2"
       ],
       "hot": [
        2,
        4
       ],
       "marks": {
        "0": "lo",
        "2": "mid",
        "4": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 3",
     "description": "1 ≤ 2 → minimum is at mid or left → hi = 3.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "4",
        "5",
        "1",
        "2"
       ],
       "hot": [
        3,
        4
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
     "description": "lo = hi = 3 → value 1.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "4",
        "5",
        "1",
        "2"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "min"
       }
      }
     ]
    }
   ]
  }
 ]
};
