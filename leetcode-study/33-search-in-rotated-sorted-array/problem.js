window.PROBLEM = {
 "id": "33",
 "title": "Search in Rotated Sorted Array",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "A sorted list was rotated (e.g. [0,1,2,4,5,6,7] became [4,5,6,7,0,1,2]). Find the target's index in O(log n).",
 "why": "target 0 in [4,5,6,7,0,1,2] → index 4.",
 "idea": "One half is always sorted. Check whether the target lies in the sorted half; if yes search there, else the other half.",
 "input": "[4,5,6,7,0,1,2], target=0",
 "output": "4",
 "notes": [],
 "best": "One half is always sorted. Check whether the target lies in the sorted half; if yes search there, else the other half.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def search(self, nums, target):\n        lo, hi = 0, len(nums) - 1\n        while lo <= hi:\n            mid = (lo + hi) // 2\n            if nums[mid] == target: return mid\n            if nums[lo] <= nums[mid]:\n                if nums[lo] <= target < nums[mid]: hi = mid - 1\n                else: lo = mid + 1\n            else:\n                if nums[mid] < target <= nums[hi]: lo = mid + 1\n                else: hi = mid - 1\n        return -1",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "One half is always sorted. Check whether the target lies in the sorted half; if yes search there, else the other half.",
   "frames": [
    {
     "title": "Start",
     "description": "Whole array.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "5",
        "6",
        "7",
        "0",
        "1",
        "2"
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
     "description": "nums[mid] = 7. Left half is sorted but target isn't in it → go right.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "5",
        "6",
        "7",
        "0",
        "1",
        "2"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "mid",
        "4": "lo",
        "6": "hi"
       }
      }
     ]
    },
    {
     "title": "mid = 5",
     "description": "nums[mid] = 1. Left half is sorted and holds the target.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "5",
        "6",
        "7",
        "0",
        "1",
        "2"
       ],
       "hot": [
        5
       ],
       "marks": {
        "4": "lo,hi",
        "5": "mid"
       }
      }
     ]
    },
    {
     "title": "mid = 4",
     "description": "Found 0 at index 4.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "5",
        "6",
        "7",
        "0",
        "1",
        "2"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "lo,hi,mid"
       }
      }
     ]
    }
   ]
  }
 ]
};
