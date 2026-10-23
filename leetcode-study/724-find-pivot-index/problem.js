window.PROBLEM = {
 "id": "724",
 "title": "Find Pivot Index",
 "level": "Easy",
 "topic": "Prefix Sum",
 "question": "Find the index where the sum on the left equals the sum on the right.",
 "input": "[1, 7, 3, 6, 5, 6]",
 "output": "3",
 "notes": [
  "No right-sum array: derive it from total.",
  "Return the leftmost pivot."
 ],
 "best": "Total once, then one pass with a running left sum.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def pivotIndex(self, nums):\n        total, left = sum(nums), 0\n        for i, x in enumerate(nums):\n            if left == total - left - x: return i\n            left += x\n        return -1",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "right = total − left − x.",
   "frames": [
    {
     "title": "Total",
     "description": "total = 28, left = 0.",
     "codeLines": [
      3
     ],
     "facts": {
      "total": 28,
      "left": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 0",
     "description": "left = 0, right = 27 → not equal",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "left": 0,
      "right": 27
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "right side",
       "cells": [
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 1",
     "description": "left = 1, right = 20 → not equal",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "left": 1,
      "right": 20
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "right side",
       "cells": [
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 2",
     "description": "left = 8, right = 17 → not equal",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "left": 8,
      "right": 17
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "right side",
       "cells": [
        "6",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 3",
     "description": "left = 11, right = 11 → equal! pivot",
     "codeLines": [
      5
     ],
     "facts": {
      "left": 11,
      "right": 11
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {
        "3": "i"
       }
      },
      {
       "name": "right side",
       "cells": [
        "5",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Pivot index = 3.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "7",
        "3",
        "6",
        "5",
        "6"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "pivot"
       }
      }
     ]
    }
   ]
  }
 ]
};
