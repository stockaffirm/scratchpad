window.PROBLEM = {
 "id": "918",
 "title": "Maximum Sum Circular Subarray",
 "level": "Medium",
 "topic": "Kadane's Algorithm",
 "question": "The list is circular (the end connects to the start). Find the biggest sum of a contiguous run that may wrap around.",
 "why": "[5,-3,5]: wrapping around, 5 + 5 = 10.",
 "idea": "The best run either doesn't wrap (normal Kadane) or wraps — then it equals total − (smallest run in the middle).",
 "input": "[5,-3,5]",
 "output": "10",
 "notes": [],
 "best": "The best run either doesn't wrap (normal Kadane) or wraps — then it equals total − (smallest run in the middle).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxSubarraySumCircular(self, nums):\n        total, cur_max, cur_min = 0, 0, 0\n        best_max, best_min = nums[0], nums[0]\n        for x in nums:\n            cur_max = max(x, cur_max + x); best_max = max(best_max, cur_max)\n            cur_min = min(x, cur_min + x); best_min = min(best_min, cur_min)\n            total += x\n        return best_max if best_max < 0 else max(best_max, total - best_min)",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "The best run either doesn't wrap (normal Kadane) or wraps — then it equals total − (smallest run in the middle).",
   "frames": [
    {
     "title": "Start",
     "description": "Track the biggest and smallest runs, and the total.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "-3",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 5",
     "description": "biggest run 5, smallest run 5, total 5",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "bestMax": 5,
      "bestMin": 5,
      "total": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "-3",
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = -3",
     "description": "biggest run 5, smallest run -3, total 2",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "bestMax": 5,
      "bestMin": -3,
      "total": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "-3",
        "5"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 5",
     "description": "biggest run 7, smallest run -3, total 7",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "bestMax": 7,
      "bestMin": -3,
      "total": 7
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "-3",
        "5"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "max(7, total 7 − smallest (−3)) = max(7, 10) = 10 (wrap-around).",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "5",
        "-3",
        "5"
       ],
       "hot": [
        2,
        0
       ],
       "marks": {}
      },
      {
       "name": "answer",
       "cells": [
        "10"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
