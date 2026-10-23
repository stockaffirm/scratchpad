window.PROBLEM = {
 "id": "34",
 "title": "Find First and Last Position of Element in Sorted Array",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "In a sorted list, find the first and last index of the target (−1, −1 if missing).",
 "why": "[5,7,7,8,8,10], target 8 → first at 3, last at 4.",
 "idea": "Binary search for the left edge (bisect_left) and the right edge (bisect_right − 1).",
 "input": "[5,7,7,8,8,10], target=8",
 "output": "[3, 4]",
 "notes": [],
 "best": "Binary search for the left edge (bisect_left) and the right edge (bisect_right − 1).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def searchRange(self, nums, target):\n        lo = bisect_left(nums, target)\n        if lo == len(nums) or nums[lo] != target: return [-1, -1]\n        return [lo, bisect_right(nums, target) - 1]",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Binary search for the left edge (bisect_left) and the right edge (bisect_right − 1).",
   "frames": [
    {
     "title": "Sorted list",
     "description": "Looking for 8.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "7",
        "7",
        "8",
        "8",
        "10"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "bisect_left",
     "description": "First position that is ≥ 8 → index 3.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "7",
        "7",
        "8",
        "8",
        "10"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "first"
       }
      }
     ]
    },
    {
     "title": "bisect_right",
     "description": "First position that is > 8 → index 5, so the last 8 is at 5 − 1 = 4.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "7",
        "7",
        "8",
        "8",
        "10"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "last"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "[3, 4]",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "3",
        "4"
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
