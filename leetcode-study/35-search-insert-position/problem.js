window.PROBLEM = {
 "id": "35",
 "title": "Search Insert Position",
 "level": "Easy",
 "topic": "Binary Search",
 "question": "In a sorted list, find the index of the target, or the index where it would be inserted to keep the list sorted.",
 "why": "[1,3,5,6], target 2 → it belongs at index 1.",
 "idea": "Binary search for the first position whose value is ≥ target.",
 "input": "[1,3,5,6], target=2",
 "output": "1",
 "notes": [],
 "best": "Binary search for the first position whose value is ≥ target.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def searchInsert(self, nums, target):\n        lo, hi = 0, len(nums)\n        while lo < hi:\n            mid = (lo + hi) // 2\n            if nums[mid] < target: lo = mid + 1\n            else: hi = mid\n        return lo",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Binary search for the first position whose value is ≥ target.",
   "frames": [
    {
     "title": "Start",
     "description": "Search range is the whole list.",
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
        "5",
        "6"
       ],
       "hot": [],
       "marks": {
        "0": "lo"
       }
      }
     ]
    },
    {
     "title": "mid = 2",
     "description": "5 ≥ 2 → hi = 2",
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
        "3",
        "5",
        "6"
       ],
       "hot": [
        2
       ],
       "marks": {
        "0": "lo",
        "2": "mid"
       }
      }
     ]
    },
    {
     "title": "mid = 1",
     "description": "3 ≥ 2 → hi = 1",
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
        "3",
        "5",
        "6"
       ],
       "hot": [
        1
       ],
       "marks": {
        "0": "lo",
        "1": "mid"
       }
      }
     ]
    },
    {
     "title": "mid = 0",
     "description": "1 < 2 → lo = 1",
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
        "3",
        "5",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "lo,mid"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Insert at index 1.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {
        "1": "insert"
       }
      }
     ]
    }
   ]
  }
 ]
};
