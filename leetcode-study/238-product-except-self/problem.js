window.PROBLEM = {
 "id": "238",
 "title": "Product of Array Except Self",
 "level": "Medium",
 "topic": "Prefix / Suffix",
 "question": "For each position, multiply every OTHER number in the list (all except itself). No division allowed.",
 "input": "[1, 2, 3, 4]",
 "output": "[24, 12, 8, 6]",
 "notes": [
  "L[i] multiplies all numbers strictly left of index i. R[i] multiplies all strictly right of it.",
  "A product of an empty set of elements is 1 (multiplication identity).",
  "The optimized version uses answer for prefix products and one variable R for running suffix.",
  "Both solutions take O(n) time; optimized uses O(1) auxiliary space excluding output."
 ],
 "best": "Running right product wins: same linear time and constant auxiliary space rather than two O(n) helper arrays.",
 "variants": [
  {
   "name": "Separate left and right arrays",
   "code": "class Solution:\n    def productExceptSelf(self, nums):\n        n = len(nums)\n        L, R, answer = [0]*n, [0]*n, [0]*n\n        L[0] = 1\n        for i in range(1,n): L[i] = nums[i-1]*L[i-1]\n        R[n-1] = 1\n        for i in reversed(range(n-1)): R[i] = nums[i+1]*R[i+1]\n        for i in range(n): answer[i] = L[i]*R[i]\n        return answer",
   "time": "O(n)",
   "space": "O(n) auxiliary",
   "explain": "Clear visualization but allocates two additional arrays.",
   "frames": [
    {
     "title": "Input",
     "description": "For each index multiply every OTHER number.",
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
        "4"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Build left products",
     "description": "No items left of index 0, so L[0] = 1.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "L",
       "cells": [
        "1",
        "1",
        "2",
        "6"
       ],
       "hot": [
        1,
        2,
        3
       ]
      }
     ]
    },
    {
     "title": "Build right products",
     "description": "No items right of index 3, so R[3] = 1.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "R",
       "cells": [
        "24",
        "12",
        "4",
        "1"
       ],
       "hot": [
        0,
        1,
        2
       ]
      }
     ]
    },
    {
     "title": "Multiply per index",
     "description": "L[i] × R[i] excludes nums[i].",
     "codeLines": [
      9,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "L",
       "cells": [
        "1",
        "1",
        "2",
        "6"
       ],
       "hot": []
      },
      {
       "name": "R",
       "cells": [
        "24",
        "12",
        "4",
        "1"
       ],
       "hot": []
      },
      {
       "name": "answer",
       "cells": [
        "24",
        "12",
        "8",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ]
      }
     ]
    }
   ]
  },
  {
   "name": "Running right product",
   "code": "class Solution:\n    def productExceptSelf(self, nums):\n        n = len(nums)\n        answer = [0] * n\n        answer[0] = 1\n        for i in range(1,n):\n            answer[i] = nums[i-1] * answer[i-1]\n        R = 1\n        for i in reversed(range(n)):\n            answer[i] = answer[i] * R\n            R *= nums[i]\n        return answer",
   "time": "O(n)",
   "space": "O(1) auxiliary; O(n) output",
   "explain": "Stores only running suffix product R, not an entire right array.",
   "frames": [
    {
     "title": "Initialize output",
     "description": "Input = [1,2,3,4], answer starts with zeros.",
     "codeLines": [
      3,
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "3",
        "4"
       ],
       "hot": []
      },
      {
       "name": "answer",
       "cells": [
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Forward pass",
     "description": "Store left products directly in answer.",
     "codeLines": [
      5,
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "1",
        "2",
        "6"
       ],
       "hot": [
        1,
        2,
        3
       ]
      }
     ]
    },
    {
     "title": "i=3: R initially 1",
     "description": "answer[3]=6×1=6, then R=4.",
     "codeLines": [
      8,
      9,
      10,
      11
     ],
     "facts": {
      "R": 4
     },
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "1",
        "2",
        "6"
       ],
       "hot": [
        3
       ]
      }
     ]
    },
    {
     "title": "i=2: R is 4",
     "description": "answer[2]=2×4=8, then R=12.",
     "codeLines": [
      9,
      10,
      11
     ],
     "facts": {
      "R": 12
     },
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "1",
        "8",
        "6"
       ],
       "hot": [
        2
       ]
      }
     ]
    },
    {
     "title": "i=1: R is 12",
     "description": "answer[1]=1×12=12, then R=24.",
     "codeLines": [
      9,
      10,
      11
     ],
     "facts": {
      "R": 24
     },
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "12",
        "8",
        "6"
       ],
       "hot": [
        1
       ]
      }
     ]
    },
    {
     "title": "i=0: R is 24",
     "description": "answer[0]=1×24=24.",
     "codeLines": [
      9,
      10,
      12
     ],
     "facts": {
      "R": 24
     },
     "rows": [
      {
       "name": "answer",
       "cells": [
        "24",
        "12",
        "8",
        "6"
       ],
       "hot": [
        0
       ]
      }
     ]
    }
   ]
  }
 ],
 "why": "[1,2,3,4]: position 0 → 2×3×4 = 24. Position 1 → 1×3×4 = 12. Position 2 → 1×2×4 = 8. Position 3 → 1×2×3 = 6.",
 "idea": "answer[i] = (product of everything left of i) × (product of everything right of i)."
};
