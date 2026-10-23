window.PROBLEM = {
 "id": "209",
 "title": "Minimum Size Subarray Sum",
 "level": "Medium",
 "topic": "Sliding Window",
 "question": "Find the shortest run of numbers in a row whose sum is at least target.",
 "why": "target 7 on [2,3,1,2,4,3]: the run [4,3] adds to 7 and has length 2.",
 "idea": "Grow the window on the right; while the sum is big enough, shrink from the left and record the length.",
 "input": "target=7, [2,3,1,2,4,3]",
 "output": "2",
 "notes": [],
 "best": "Grow the window on the right; while the sum is big enough, shrink from the left and record the length.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minSubArrayLen(self, target, nums):\n        l = total = 0; best = float('inf')\n        for r, x in enumerate(nums):\n            total += x\n            while total >= target:\n                best = min(best, r - l + 1); total -= nums[l]; l += 1\n        return 0 if best == float('inf') else best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Grow the window on the right; while the sum is big enough, shrink from the left and record the length.",
   "frames": [
    {
     "title": "Start",
     "description": "Window empty.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "r = 0",
     "description": "sum = 2",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "sum": 2,
      "best": "∞"
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "l,r"
       }
      }
     ]
    },
    {
     "title": "r = 1",
     "description": "sum = 5",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "sum": 5,
      "best": "∞"
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "0": "l",
        "1": "r"
       }
      }
     ]
    },
    {
     "title": "r = 2",
     "description": "sum = 6",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "sum": 6,
      "best": "∞"
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {
        "0": "l",
        "2": "r"
       }
      }
     ]
    },
    {
     "title": "r = 3",
     "description": "sum = 8 ≥ 7 → record length 4, shrink",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "sum": 6,
      "best": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [
        1,
        2,
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
     "title": "r = 4",
     "description": "sum = 10 ≥ 7 → record length 4, shrink ≥ 7 → record length 3, shrink",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "sum": 6,
      "best": 3
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {
        "3": "l",
        "4": "r"
       }
      }
     ]
    },
    {
     "title": "r = 5",
     "description": "sum = 9 ≥ 7 → record length 3, shrink ≥ 7 → record length 2, shrink",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "sum": 3,
      "best": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "2",
        "4",
        "3"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "l,r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Shortest = 2.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "2"
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
