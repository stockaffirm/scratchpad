window.PROBLEM = {
 "id": "27",
 "title": "Remove Element",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Remove every copy of val from the list in place. Return how many items remain; the first k slots must hold them.",
 "why": "[3,2,2,3], val = 3 → keep 2,2 → return 2.",
 "idea": "One pointer reads, another writes: copy only the values you want to keep.",
 "input": "nums=[3,2,2,3], val=3",
 "output": "2 → [2,2,_,_]",
 "notes": [],
 "best": "One pointer reads, another writes: copy only the values you want to keep.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def removeElement(self, nums, val):\n        k = 0\n        for x in nums:\n            if x != val:\n                nums[k] = x; k += 1\n        return k",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "One pointer reads, another writes: copy only the values you want to keep.",
   "frames": [
    {
     "title": "Start",
     "description": "k = next free slot.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
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
     "title": "x = 3",
     "description": "Equals val → skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {
        "0": "k,read"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Not 3 → keep it at k.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "2",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {
        "1": "k,read"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Not 3 → keep it at k.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "2",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {
        "2": "k,read"
       }
      }
     ]
    },
    {
     "title": "x = 3",
     "description": "Equals val → skip.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {
        "2": "k",
        "3": "read"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "First k = 2 slots hold the answer.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
