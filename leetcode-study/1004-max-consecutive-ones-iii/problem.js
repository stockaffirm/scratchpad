window.PROBLEM = {
 "id": "1004",
 "title": "Max Consecutive Ones III",
 "level": "Medium",
 "topic": "Sliding Window",
 "question": "A list of 0s and 1s. You may turn at most k zeros into ones. What's the longest run of 1s you can make?",
 "input": "[1,0,1,1,0,1], k=1",
 "output": "4",
 "notes": [
  "The window never holds more than k zeros.",
  "left only moves forward, so it is O(n) total."
 ],
 "best": "Variable window: expand always, shrink only when invalid.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestOnes(self, nums, k):\n        left = zeros = best = 0\n        for right, x in enumerate(nums):\n            if x == 0: zeros += 1\n            while zeros > k:\n                if nums[left] == 0: zeros -= 1\n                left += 1\n            best = max(best, right - left + 1)\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Grow right; shrink left while too many zeros.",
   "frames": [
    {
     "title": "Input",
     "description": "Allow at most 1 zero in the window.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
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
     "description": "Window is valid.",
     "codeLines": [
      5,
      9
     ],
     "facts": {
      "zeros": 0,
      "best": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
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
     "description": "Window is valid.",
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
     "description": "Window is valid.",
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
     "description": "Window is valid.",
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
     "description": "Too many zeros → shrink left to 2.",
     "codeLines": [
      5,
      6,
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
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        2,
        3,
        4
       ],
       "marks": {
        "2": "l",
        "4": "r"
       }
      }
     ],
     "sub": [
      {
       "description": "add nums[4] = 0 → zeros = 2",
       "codeLines": [
        5
       ],
       "facts": {
        "zeros": 2
       },
       "rows": [
        {
         "name": "nums",
         "cells": [
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
          3,
          4
         ],
         "marks": {
          "0": "l",
          "4": "r"
         }
        }
       ]
      },
      {
       "description": "zeros > 1 → shrink: nums[0] is 1 → left moves to 1",
       "codeLines": [
        6
       ],
       "facts": {
        "zeros": 2
       },
       "rows": [
        {
         "name": "nums",
         "cells": [
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
          3,
          4
         ],
         "marks": {
          "0": "l",
          "4": "r"
         }
        }
       ]
      },
      {
       "description": "zeros > 1 → shrink: nums[1] is 0 → zeros = 1, left moves to 2",
       "codeLines": [
        6
       ],
       "facts": {
        "zeros": 1
       },
       "rows": [
        {
         "name": "nums",
         "cells": [
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
       "description": "zeros ≤ 1 → window is valid again: length 3",
       "codeLines": [
        9
       ],
       "facts": {
        "zeros": 1
       },
       "rows": [
        {
         "name": "nums",
         "cells": [
          "1",
          "0",
          "1",
          "1",
          "0",
          "1"
         ],
         "hot": [
          2,
          3,
          4
         ],
         "marks": {
          "2": "l",
          "4": "r"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "right = 5",
     "description": "Window is valid.",
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
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
       ],
       "hot": [
        2,
        3,
        4,
        5
       ],
       "marks": {
        "2": "l",
        "5": "r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Longest valid window = 4.",
     "codeLines": [
      10
     ],
     "facts": {
      "best": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "0",
        "1",
        "1",
        "0",
        "1"
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
    }
   ]
  }
 ],
 "why": "k=1 on [1,0,1,1,0,1]: flip the second 0 and positions 2–5 become 1,1,1,1 → length 4.",
 "idea": "Keep a window with at most k zeros. Grow on the right; shrink on the left when there are too many zeros."
};
