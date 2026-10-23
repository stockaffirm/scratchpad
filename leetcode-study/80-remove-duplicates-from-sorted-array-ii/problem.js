window.PROBLEM = {
 "id": "80",
 "title": "Remove Duplicates from Sorted Array II",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Sorted list. Allow each value at most TWICE. Remove extras in place and return the new length.",
 "why": "[1,1,1,2,2,3] → 1,1,2,2,3 → length 5.",
 "idea": "Keep x if fewer than 2 are kept, or if it differs from the value two places back.",
 "input": "[1,1,1,2,2,3]",
 "output": "5 → [1,1,2,2,3,_]",
 "notes": [],
 "best": "Keep x if fewer than 2 are kept, or if it differs from the value two places back.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def removeDuplicates(self, nums):\n        k = 0\n        for x in nums:\n            if k < 2 or x != nums[k - 2]:\n                nums[k] = x; k += 1\n        return k",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Keep x if fewer than 2 are kept, or if it differs from the value two places back.",
   "frames": [
    {
     "title": "Start",
     "description": "Allow up to two copies.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "1",
        "2",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {
        "0": "k"
       }
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Fewer than 2 copies kept → keep.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "1",
        "2",
        "2",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "read",
        "1": "k"
       }
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Fewer than 2 copies kept → keep.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "1",
        "2",
        "2",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "read",
        "2": "k"
       }
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Already two copies of 1 → skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "1",
        "2",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {
        "2": "k,read"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Fewer than 2 copies kept → keep.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "2",
        "2",
        "2",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {
        "3": "k,read"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Fewer than 2 copies kept → keep.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "2",
        "2",
        "2",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {
        "4": "k,read"
       }
      }
     ]
    },
    {
     "title": "x = 3",
     "description": "Fewer than 2 copies kept → keep.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "2",
        "2",
        "3",
        "3"
       ],
       "hot": [
        4
       ],
       "marks": {
        "5": "k,read"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Length 5.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "1",
        "2",
        "2",
        "3"
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
