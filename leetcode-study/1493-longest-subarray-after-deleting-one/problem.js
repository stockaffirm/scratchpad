window.PROBLEM = {
 "id": "1493",
 "title": "Longest Subarray of 1's After Deleting One Element",
 "level": "Medium",
 "topic": "Sliding Window",
 "question": "Delete exactly one element; longest run of 1s left?",
 "input": "[0,1,1,1,0,1,1,0,1]",
 "output": "5",
 "notes": [
  "right - left (not +1) accounts for the deleted element.",
  "You must delete one even if all are 1s."
 ],
 "best": "Same window as 1004 with k = 1, minus the deleted element.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestSubarray(self, nums):\n        left = zeros = best = 0\n        for right, x in enumerate(nums):\n            if x == 0: zeros += 1\n            while zeros > 1:\n                if nums[left] == 0: zeros -= 1\n                left += 1\n            best = max(best, right - left)\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Window with at most one zero; the deleted slot costs 1 (right - left).",
   "frames": [
    {
     "title": "Input",
     "description": "Window may hold at most one zero (the one we delete).",
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
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [],
       "marks": {
        "0": "l,r"
       }
      }
     ]
    },
    {
     "title": "right = 0",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
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
     "title": "right = 1",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
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
     "title": "right = 2",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
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
     "title": "right = 3",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 3
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {
        "0": "l",
        "3": "r"
       }
      }
     ]
    },
    {
     "title": "right = 4",
     "description": "Two zeros → shrink left to 1.",
     "codeLines": [
      5,
      6,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 3
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        1,
        2,
        3,
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
     "title": "right = 5",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {
        "1": "l",
        "5": "r"
       }
      }
     ]
    },
    {
     "title": "right = 6",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        1,
        2,
        3,
        4,
        5,
        6
       ],
       "marks": {
        "1": "l",
        "6": "r"
       }
      }
     ]
    },
    {
     "title": "right = 7",
     "description": "Two zeros → shrink left to 5.",
     "codeLines": [
      5,
      6,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        5,
        6,
        7
       ],
       "marks": {
        "5": "l",
        "7": "r"
       }
      }
     ]
    },
    {
     "title": "right = 8",
     "description": "Valid window.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 1,
      "best": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        5,
        6,
        7,
        8
       ],
       "marks": {
        "5": "l",
        "8": "r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Best = 5.",
     "codeLines": [
      10
     ],
     "facts": {
      "best": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "1",
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        3,
        4,
        5,
        6,
        7
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
