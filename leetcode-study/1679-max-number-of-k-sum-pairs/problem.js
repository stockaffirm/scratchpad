window.PROBLEM = {
 "id": "1679",
 "title": "Max Number of K-Sum Pairs",
 "level": "Medium",
 "topic": "Two Pointers",
 "question": "Repeatedly remove two numbers that add to k. How many operations?",
 "input": "[3, 1, 3, 4, 3], k=6",
 "output": "1",
 "notes": [
  "Sum too small → move left up; too big → move right down.",
  "Each number is used at most once."
 ],
 "best": "Sort + two pointers; a hash-map count gives O(n) if order doesn't matter.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxOperations(self, nums, k):\n        nums.sort()\n        l, r, ops = 0, len(nums) - 1, 0\n        while l < r:\n            s = nums[l] + nums[r]\n            if s == k:\n                ops += 1; l += 1; r -= 1\n            elif s < k: l += 1\n            else: r -= 1\n        return ops",
   "time": "O(n log n)",
   "space": "O(1)",
   "explain": "Sort, then squeeze from both ends.",
   "frames": [
    {
     "title": "Sorted",
     "description": "Sort first: [1,3,3,3,4], k = 6.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "3",
        "3",
        "4"
       ],
       "hot": [],
       "marks": {
        "0": "l",
        "4": "r"
       }
      }
     ]
    },
    {
     "title": "sum = 5",
     "description": "1 + 4 < k → move l.",
     "codeLines": [
      6,
      9
     ],
     "facts": {
      "ops": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "3",
        "3",
        "4"
       ],
       "hot": [
        0,
        4
       ],
       "marks": {
        "0": "l",
        "4": "r"
       }
      }
     ]
    },
    {
     "title": "sum = 7",
     "description": "3 + 4 > k → move r.",
     "codeLines": [
      6,
      10
     ],
     "facts": {
      "ops": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "3",
        "3",
        "4"
       ],
       "hot": [
        1,
        4
       ],
       "marks": {
        "1": "l",
        "4": "r"
       }
      }
     ]
    },
    {
     "title": "sum = 6",
     "description": "3 + 3 = k → pair! ops = 1.",
     "codeLines": [
      6,
      8
     ],
     "facts": {
      "ops": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "3",
        "3",
        "4"
       ],
       "hot": [
        1,
        3
       ],
       "marks": {
        "1": "l",
        "3": "r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "l met r.",
     "codeLines": [
      11
     ],
     "facts": {
      "ops": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "3",
        "3",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
