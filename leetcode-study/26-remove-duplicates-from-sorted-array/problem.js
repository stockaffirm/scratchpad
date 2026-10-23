window.PROBLEM = {
 "id": "26",
 "title": "Remove Duplicates from Sorted Array",
 "level": "Easy",
 "topic": "Array / String",
 "question": "The list is sorted. Remove duplicates in place so each value appears once. Return how many unique values.",
 "why": "[0,0,1,1,1,2,2,3,3,4] → 0,1,2,3,4 → return 5.",
 "idea": "Compare each number with the last kept one; keep it only if different.",
 "input": "[0,0,1,1,1,2,2,3,3,4]",
 "output": "5 → [0,1,2,3,4,…]",
 "notes": [],
 "best": "Compare each number with the last kept one; keep it only if different.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def removeDuplicates(self, nums):\n        k = 1\n        for i in range(1, len(nums)):\n            if nums[i] != nums[k - 1]:\n                nums[k] = nums[i]; k += 1\n        return k",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Compare each number with the last kept one; keep it only if different.",
   "frames": [
    {
     "title": "Start",
     "description": "The first number is always kept; k = 1.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "0",
        "1",
        "1",
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {
        "1": "k"
       }
      }
     ]
    },
    {
     "title": "i = 1",
     "description": "0 is a repeat → skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "0",
        "1",
        "1",
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "k,i"
       }
      }
     ]
    },
    {
     "title": "i = 2",
     "description": "1 is new → write at slot 1.",
     "codeLines": [
      5,
      6
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
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {
        "2": "k,i"
       }
      }
     ]
    },
    {
     "title": "i = 3",
     "description": "1 is a repeat → skip.",
     "codeLines": [
      5
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
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        3
       ],
       "marks": {
        "2": "k",
        "3": "i"
       }
      }
     ]
    },
    {
     "title": "i = 4",
     "description": "1 is a repeat → skip.",
     "codeLines": [
      5
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
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        4
       ],
       "marks": {
        "2": "k",
        "4": "i"
       }
      }
     ]
    },
    {
     "title": "i = 5",
     "description": "2 is new → write at slot 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "1",
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        2,
        5
       ],
       "marks": {
        "3": "k",
        "5": "i"
       }
      }
     ]
    },
    {
     "title": "i = 6",
     "description": "2 is a repeat → skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "1",
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        6
       ],
       "marks": {
        "3": "k",
        "6": "i"
       }
      }
     ]
    },
    {
     "title": "i = 7",
     "description": "3 is new → write at slot 3.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        3,
        7
       ],
       "marks": {
        "4": "k",
        "7": "i"
       }
      }
     ]
    },
    {
     "title": "i = 8",
     "description": "3 is a repeat → skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "1",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        8
       ],
       "marks": {
        "4": "k",
        "8": "i"
       }
      }
     ]
    },
    {
     "title": "i = 9",
     "description": "4 is new → write at slot 4.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "2",
        "2",
        "3",
        "3",
        "4"
       ],
       "hot": [
        4,
        9
       ],
       "marks": {
        "5": "k",
        "9": "i"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "5 unique values.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4"
       ],
       "hot": [
        0,
        1,
        2,
        3,
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
