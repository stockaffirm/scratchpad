window.PROBLEM = {
 "id": "42",
 "title": "Trapping Rain Water",
 "level": "Hard",
 "topic": "Array / String",
 "question": "Bars of different heights. After rain, how much water is trapped between the bars?",
 "why": "[0,1,0,2,1,0,1,3,2,1,2,1] traps 6 units.",
 "idea": "Water above a bar = (shorter of the tallest bar on each side) − its height. Use two pointers moving inward.",
 "input": "[0,1,0,2,1,0,1,3,2,1,2,1]",
 "output": "6",
 "notes": [],
 "best": "Water above a bar = (shorter of the tallest bar on each side) − its height. Use two pointers moving inward.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def trap(self, height):\n        l, r = 0, len(height) - 1\n        lmax = rmax = water = 0\n        while l < r:\n            if height[l] < height[r]:\n                lmax = max(lmax, height[l]); water += lmax - height[l]; l += 1\n            else:\n                rmax = max(rmax, height[r]); water += rmax - height[r]; r -= 1\n        return water",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Water above a bar = (shorter of the tallest bar on each side) − its height. Use two pointers moving inward.",
   "frames": [
    {
     "title": "Start",
     "description": "Two pointers at both ends.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [],
       "marks": {
        "0": "l",
        "11": "r"
       }
      }
     ]
    },
    {
     "title": "l=0, r=11",
     "description": "Left is lower: left max 0, water +0",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 0
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "l",
        "11": "r"
       }
      }
     ]
    },
    {
     "title": "l=1, r=11",
     "description": "Right is lower: right max 1, water +0",
     "codeLines": [
      6,
      9
     ],
     "facts": {
      "water": 0
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        11
       ],
       "marks": {
        "1": "l",
        "11": "r"
       }
      }
     ]
    },
    {
     "title": "l=1, r=10",
     "description": "Left is lower: left max 1, water +0",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 0
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "l",
        "10": "r"
       }
      }
     ]
    },
    {
     "title": "l=2, r=10",
     "description": "Left is lower: left max 1, water +1",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 1
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "l",
        "10": "r"
       }
      }
     ]
    },
    {
     "title": "l=3, r=10",
     "description": "Right is lower: right max 2, water +0",
     "codeLines": [
      6,
      9
     ],
     "facts": {
      "water": 1
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        10
       ],
       "marks": {
        "3": "l",
        "10": "r"
       }
      }
     ]
    },
    {
     "title": "l=3, r=9",
     "description": "Right is lower: right max 2, water +1",
     "codeLines": [
      6,
      9
     ],
     "facts": {
      "water": 2
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        9
       ],
       "marks": {
        "3": "l",
        "9": "r"
       }
      }
     ]
    },
    {
     "title": "l=3, r=8",
     "description": "Right is lower: right max 2, water +0",
     "codeLines": [
      6,
      9
     ],
     "facts": {
      "water": 2
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        8
       ],
       "marks": {
        "3": "l",
        "8": "r"
       }
      }
     ]
    },
    {
     "title": "l=3, r=7",
     "description": "Left is lower: left max 2, water +0",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 2
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "l",
        "7": "r"
       }
      }
     ]
    },
    {
     "title": "l=4, r=7",
     "description": "Left is lower: left max 2, water +1",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 3
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "l",
        "7": "r"
       }
      }
     ]
    },
    {
     "title": "l=5, r=7",
     "description": "Left is lower: left max 2, water +2",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 5
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "l",
        "7": "r"
       }
      }
     ]
    },
    {
     "title": "l=6, r=7",
     "description": "Left is lower: left max 2, water +1",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "water": 6
     },
     "rows": [
      {
       "name": "height",
       "cells": [
        "0",
        "1",
        "0",
        "2",
        "1",
        "0",
        "1",
        "3",
        "2",
        "1",
        "2",
        "1"
       ],
       "hot": [
        6
       ],
       "marks": {
        "6": "l",
        "7": "r"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "6 units trapped.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "6"
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
