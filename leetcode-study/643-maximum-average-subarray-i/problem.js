window.PROBLEM = {
 "id": "643",
 "title": "Maximum Average Subarray I",
 "level": "Easy",
 "topic": "Sliding Window",
 "question": "Find the contiguous subarray of length k with the largest average.",
 "input": "[1, 12, -5, -6, 50, 3], k=4",
 "output": "12.75",
 "notes": [
  "Compare sums, divide once at the end.",
  "Never recompute the whole window."
 ],
 "best": "Sliding window updates the sum in O(1) per step.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findMaxAverage(self, nums, k):\n        window = sum(nums[:k])\n        best = window\n        for i in range(k, len(nums)):\n            window += nums[i] - nums[i - k]\n            best = max(best, window)\n        return best / k",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Slide the window: add the new element, drop the old one.",
   "frames": [
    {
     "title": "First window",
     "description": "Sum of the first 4 = 2.",
     "codeLines": [
      3
     ],
     "facts": {
      "window": 2,
      "best": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "12",
        "-5",
        "-6",
        "50",
        "3"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slide to i=4",
     "description": "Add 50, drop 1.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "window": 51,
      "best": 51
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "12",
        "-5",
        "-6",
        "50",
        "3"
       ],
       "hot": [
        1,
        2,
        3,
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slide to i=5",
     "description": "Add 3, drop 12.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "window": 42,
      "best": 51
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "12",
        "-5",
        "-6",
        "50",
        "3"
       ],
       "hot": [
        2,
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "51 / 4 = 12.75",
     "codeLines": [
      8
     ],
     "facts": {
      "answer": 12.75
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "12",
        "-5",
        "-6",
        "50",
        "3"
       ],
       "hot": [
        1,
        2,
        3,
        4
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
