window.PROBLEM = {
 "id": "283",
 "title": "Move Zeroes",
 "level": "Easy",
 "topic": "Two Pointers",
 "question": "Move all zeroes to the end in place, keeping the order of non-zero values.",
 "input": "[0, 1, 0, 3, 12]",
 "output": "[1, 3, 12, 0, 0]",
 "notes": [
  "slow always points at the first zero (or the next free slot).",
  "Swapping keeps zeros moving right."
 ],
 "best": "A single pass with a slow write pointer; no extra array.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def moveZeroes(self, nums):\n        slow = 0\n        for fast in range(len(nums)):\n            if nums[fast] != 0:\n                nums[slow], nums[fast] = nums[fast], nums[slow]\n                slow += 1",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "slow marks where the next non-zero belongs.",
   "frames": [
    {
     "title": "Input",
     "description": "slow = 0; fast scans.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "0",
        "3",
        "12"
       ],
       "hot": [],
       "marks": {
        "0": "slow"
       }
      }
     ]
    },
    {
     "title": "fast = 0",
     "description": "Zero: skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "0",
        "3",
        "12"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "slow,fast"
       }
      }
     ]
    },
    {
     "title": "fast = 1",
     "description": "Non-zero: swap into slow, then slow++.",
     "codeLines": [
      5,
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "0",
        "0",
        "3",
        "12"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "1": "slow,fast"
       }
      }
     ]
    },
    {
     "title": "fast = 2",
     "description": "Zero: skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "0",
        "0",
        "3",
        "12"
       ],
       "hot": [
        2
       ],
       "marks": {
        "1": "slow",
        "2": "fast"
       }
      }
     ]
    },
    {
     "title": "fast = 3",
     "description": "Non-zero: swap into slow, then slow++.",
     "codeLines": [
      5,
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "0",
        "0",
        "12"
       ],
       "hot": [
        1,
        3
       ],
       "marks": {
        "2": "slow",
        "3": "fast"
       }
      }
     ]
    },
    {
     "title": "fast = 4",
     "description": "Non-zero: swap into slow, then slow++.",
     "codeLines": [
      5,
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "3",
        "12",
        "0",
        "0"
       ],
       "hot": [
        2,
        4
       ],
       "marks": {
        "3": "slow",
        "4": "fast"
       }
      }
     ]
    }
   ]
  }
 ]
};
