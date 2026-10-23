window.PROBLEM = {
 "id": "1",
 "title": "Two Sum",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Find two numbers in the list that add up to the target and return their positions.",
 "why": "[2,7,11,15], target 9: 2 + 7 = 9 → positions 0 and 1.",
 "idea": "For each number, check if (target − number) was already seen, using a dictionary.",
 "input": "[2,7,11,15], target=9",
 "output": "[0, 1]",
 "notes": [],
 "best": "For each number, check if (target − number) was already seen, using a dictionary.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in seen: return [seen[target - x], i]\n            seen[x] = i",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "For each number, check if (target − number) was already seen, using a dictionary.",
   "frames": [
    {
     "title": "Start",
     "description": "seen is empty.",
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
        "11",
        "15"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Need 7 — not seen yet. Remember 2.",
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
        "7",
        "11",
        "15"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "seen",
       "cells": [
        "2→0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 7",
     "description": "Need 2 — seen at index 0!",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "7",
        "11",
        "15"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      },
      {
       "name": "seen",
       "cells": [
        "2→0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "[0, 1]",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "0",
        "1"
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
