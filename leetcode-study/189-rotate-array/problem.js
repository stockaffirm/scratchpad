window.PROBLEM = {
 "id": "189",
 "title": "Rotate Array",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Rotate the list to the right by k steps, in place.",
 "why": "[1..7], k = 3: the last 3 numbers move to the front → [5,6,7,1,2,3,4].",
 "idea": "Reverse everything, then reverse the first k, then reverse the rest.",
 "input": "[1,2,3,4,5,6,7], k=3",
 "output": "[5,6,7,1,2,3,4]",
 "notes": [],
 "best": "Reverse everything, then reverse the first k, then reverse the rest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def rotate(self, nums, k):\n        k %= len(nums)\n        def rev(a, b):\n            while a < b:\n                nums[a], nums[b] = nums[b], nums[a]; a += 1; b -= 1\n        rev(0, len(nums) - 1)\n        rev(0, k - 1)\n        rev(k, len(nums) - 1)",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Reverse everything, then reverse the first k, then reverse the rest.",
   "frames": [
    {
     "title": "Start",
     "description": "k = 3.",
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
        "3",
        "4",
        "5",
        "6",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Reverse all",
     "description": "Whole list flipped.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
        "1"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Reverse first k",
     "description": "First 3 flipped back: 5,6,7.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "6",
        "7",
        "4",
        "3",
        "2",
        "1"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Reverse the rest",
     "description": "Last 4 flipped back: 1,2,3,4.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "6",
        "7",
        "1",
        "2",
        "3",
        "4"
       ],
       "hot": [
        3,
        4,
        5,
        6
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
