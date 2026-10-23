window.PROBLEM = {
 "id": "198",
 "title": "House Robber",
 "level": "Medium",
 "topic": "DP - 1D",
 "question": "Rob houses for maximum money without robbing two adjacent houses.",
 "input": "[2, 7, 9, 3, 1]",
 "output": "12",
 "notes": [
  "Each house: take it or leave it.",
  "Only the previous two best values are needed."
 ],
 "best": "Rolling DP: best(i) = max(best(i-1), best(i-2) + x).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def rob(self, nums):\n        prev = cur = 0\n        for x in nums:\n            prev, cur = cur, max(cur, prev + x)\n        return cur",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Skip this house or rob it plus the best from two back.",
   "frames": [
    {
     "title": "Start",
     "description": "best so far = 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "best",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "House 0 (2)",
     "description": "max(skip: 0, rob: 2) → 2",
     "codeLines": [
      5
     ],
     "facts": {
      "prev": 0,
      "cur": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "best",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "House 1 (7)",
     "description": "max(skip: 2, rob: 7) → 7",
     "codeLines": [
      5
     ],
     "facts": {
      "prev": 2,
      "cur": 7
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "best",
       "cells": [
        "2",
        "7"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "House 2 (9)",
     "description": "max(skip: 7, rob: 11) → 11",
     "codeLines": [
      5
     ],
     "facts": {
      "prev": 7,
      "cur": 11
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "best",
       "cells": [
        "2",
        "7",
        "11"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "House 3 (3)",
     "description": "max(skip: 11, rob: 10) → 11",
     "codeLines": [
      5
     ],
     "facts": {
      "prev": 11,
      "cur": 11
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "best",
       "cells": [
        "2",
        "7",
        "11",
        "11"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "House 4 (1)",
     "description": "max(skip: 11, rob: 12) → 12",
     "codeLines": [
      5
     ],
     "facts": {
      "prev": 11,
      "cur": 12
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "best",
       "cells": [
        "2",
        "7",
        "11",
        "11",
        "12"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Max = 12 (rob 2, 9, 1).",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "9",
        "3",
        "1"
       ],
       "hot": [
        0,
        2,
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
