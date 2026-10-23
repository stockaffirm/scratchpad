window.PROBLEM = {
 "id": "53",
 "title": "Maximum Subarray",
 "level": "Medium",
 "topic": "Kadane's Algorithm",
 "question": "Find the contiguous run of numbers with the biggest sum.",
 "why": "[-2,1,-3,4,-1,2,1,-5,4]: the run [4,-1,2,1] sums to 6.",
 "idea": "Kadane: at each number, either extend the current run or start fresh from this number.",
 "input": "[-2,1,-3,4,-1,2,1,-5,4]",
 "output": "6",
 "notes": [],
 "best": "Kadane: at each number, either extend the current run or start fresh from this number.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxSubArray(self, nums):\n        cur = best = nums[0]\n        for x in nums[1:]:\n            cur = max(x, cur + x); best = max(best, cur)\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Kadane: at each number, either extend the current run or start fresh from this number.",
   "frames": [
    {
     "title": "Start",
     "description": "cur = best = first number.",
     "codeLines": [
      3
     ],
     "facts": {
      "cur": -2,
      "best": -2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Start fresh at 1",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 1,
      "best": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = -3",
     "description": "Extend → -2",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": -2,
      "best": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 4",
     "description": "Start fresh at 4",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 4,
      "best": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = -1",
     "description": "Extend → 3",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 3,
      "best": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Extend → 5",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 5,
      "best": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Extend → 6",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 6,
      "best": 6
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        6
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = -5",
     "description": "Extend → 1",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 1,
      "best": 6
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        7
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 4",
     "description": "Extend → 5",
     "codeLines": [
      5
     ],
     "facts": {
      "cur": 5,
      "best": 6
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
        "4"
       ],
       "hot": [
        8
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Best = 6 (run 4, −1, 2, 1).",
     "codeLines": [
      6
     ],
     "facts": {
      "best": 6
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "-2",
        "1",
        "-3",
        "4",
        "-1",
        "2",
        "1",
        "-5",
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
