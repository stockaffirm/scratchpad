window.PROBLEM = {
 "id": "15",
 "title": "3Sum",
 "level": "Medium",
 "topic": "Two Pointers",
 "question": "Find all unique triples in the list that add up to 0.",
 "why": "[-1,0,1,2,-1,-4] → [-1,-1,2] and [-1,0,1].",
 "idea": "Sort. Fix one number, then use two pointers on the rest; skip duplicates.",
 "input": "[-1,0,1,2,-1,-4]",
 "output": "[[-1,-1,2],[-1,0,1]]",
 "notes": [],
 "best": "Sort. Fix one number, then use two pointers on the rest; skip duplicates.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def threeSum(self, nums):\n        nums.sort(); res = []\n        for i in range(len(nums) - 2):\n            if i and nums[i] == nums[i-1]: continue\n            l, r = i + 1, len(nums) - 1\n            while l < r:\n                s = nums[i] + nums[l] + nums[r]\n                if s < 0: l += 1\n                elif s > 0: r -= 1\n                else:\n                    res.append([nums[i], nums[l], nums[r]]); l += 1\n                    while l < r and nums[l] == nums[l-1]: l += 1\n        return res",
   "time": "O(n²)",
   "space": "O(1)",
   "explain": "Sort. Fix one number, then use two pointers on the rest; skip duplicates.",
   "frames": [
    {
     "title": "Sorted",
     "description": "Fix i, then squeeze l and r.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-4",
        "-1",
        "-1",
        "0",
        "1",
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = -4",
     "description": "-4 + (-1) + 2 = -3 < 0 → l moves right … no triple works with -4.",
     "codeLines": [
      9
     ],
     "facts": {
      "sum": -3
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-4",
        "-1",
        "-1",
        "0",
        "1",
        "2"
       ],
       "hot": [
        0,
        1,
        5
       ],
       "marks": {
        "0": "i",
        "1": "l",
        "5": "r"
       }
      }
     ]
    },
    {
     "title": "i = -1 (first)",
     "description": "-1 + (-1) + 2 = 0 → save [-1,-1,2].",
     "codeLines": [
      12
     ],
     "facts": {
      "sum": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-4",
        "-1",
        "-1",
        "0",
        "1",
        "2"
       ],
       "hot": [
        1,
        2,
        5
       ],
       "marks": {
        "1": "i",
        "2": "l",
        "5": "r"
       }
      }
     ]
    },
    {
     "title": "Next",
     "description": "-1 + 0 + 1 = 0 → save [-1,0,1].",
     "codeLines": [
      12
     ],
     "facts": {
      "sum": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-4",
        "-1",
        "-1",
        "0",
        "1",
        "2"
       ],
       "hot": [
        1,
        3,
        4
       ],
       "marks": {
        "1": "i",
        "3": "l",
        "4": "r"
       }
      }
     ]
    },
    {
     "title": "i = -1 (second)",
     "description": "Same as previous i → skipped (no duplicates).",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-4",
        "-1",
        "-1",
        "0",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "i"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Two unique triples.",
     "codeLines": [
      14
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "[-1,-1,2]",
        "[-1,0,1]"
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
