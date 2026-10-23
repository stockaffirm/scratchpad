window.PROBLEM = {
 "id": "11",
 "title": "Container With Most Water",
 "level": "Medium",
 "topic": "Two Pointers",
 "question": "Pick two lines that, with the x-axis, hold the most water.",
 "input": "[1, 8, 6, 2, 5, 4, 8, 3, 7]",
 "output": "49",
 "notes": [
  "Area = min(height) × width.",
  "Width shrinks every step, so only a taller short-wall can improve."
 ],
 "best": "Two pointers: moving the taller wall can never help, so move the shorter one.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxArea(self, height):\n        l, r = 0, len(height) - 1\n        best = 0\n        while l < r:\n            best = max(best, min(height[l], height[r]) * (r - l))\n            if height[l] < height[r]: l += 1\n            else: r -= 1\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Always move the shorter wall inward.",
   "frames": [
    {
     "title": "Input",
     "description": "Start with the widest container.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        0,
        8
       ],
       "marks": {
        "0": "l",
        "8": "r"
       }
      }
     ]
    },
    {
     "title": "l=0, r=8",
     "description": "area = 1 × 8 = 8. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 8,
      "best": 8
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        0,
        8
       ],
       "marks": {
        "0": "l",
        "8": "r"
       }
      }
     ]
    },
    {
     "title": "l=1, r=8",
     "description": "area = 7 × 7 = 49. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 49,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        1,
        8
       ],
       "marks": {
        "1": "l",
        "8": "r"
       }
      }
     ]
    },
    {
     "title": "l=1, r=7",
     "description": "area = 3 × 6 = 18. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 18,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        1,
        7
       ],
       "marks": {
        "1": "l",
        "7": "r"
       }
      }
     ]
    },
    {
     "title": "l=1, r=6",
     "description": "area = 8 × 5 = 40. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 40,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        1,
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
     "title": "l=1, r=5",
     "description": "area = 4 × 4 = 16. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 16,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        1,
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
     "title": "l=1, r=4",
     "description": "area = 5 × 3 = 15. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 15,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
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
     "title": "l=1, r=3",
     "description": "area = 2 × 2 = 4. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 4,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
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
     "title": "l=1, r=2",
     "description": "area = 6 × 1 = 6. Move the shorter wall.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "area": 6,
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {
        "1": "l",
        "2": "r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Pointers met.",
     "codeLines": [
      9
     ],
     "facts": {
      "best": 49
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "1",
        "8",
        "6",
        "2",
        "5",
        "4",
        "8",
        "3",
        "7"
       ],
       "hot": [
        1,
        8
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
