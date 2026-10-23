window.PROBLEM = {
 "id": "45",
 "title": "Jump Game II",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Same jump rules, and the end is always reachable. What's the fewest jumps to reach the last spot?",
 "why": "[2,3,1,1,4]: jump 0→1 (1 jump), then 1→4 (2 jumps) → 2.",
 "idea": "Think in 'waves': when you reach the end of the current wave, take a jump to the farthest spot seen.",
 "input": "[2,3,1,1,4]",
 "output": "2",
 "notes": [],
 "best": "Think in 'waves': when you reach the end of the current wave, take a jump to the farthest spot seen.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def jump(self, nums):\n        jumps = cur_end = far = 0\n        for i in range(len(nums) - 1):\n            far = max(far, i + nums[i])\n            if i == cur_end:\n                jumps += 1; cur_end = far\n        return jumps",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Think in 'waves': when you reach the end of the current wave, take a jump to the farthest spot seen.",
   "frames": [
    {
     "title": "Start",
     "description": "No jumps yet.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "1",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "end"
       }
      }
     ]
    },
    {
     "title": "i = 0",
     "description": "farthest = 2 · end of wave → jump! next wave ends at 2",
     "codeLines": [
      5,
      7
     ],
     "facts": {
      "jumps": 1,
      "far": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "1",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "i",
        "2": "end"
       }
      }
     ]
    },
    {
     "title": "i = 1",
     "description": "farthest = 4",
     "codeLines": [
      5
     ],
     "facts": {
      "jumps": 1,
      "far": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "1",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "i",
        "2": "end"
       }
      }
     ]
    },
    {
     "title": "i = 2",
     "description": "farthest = 4 · end of wave → jump! next wave ends at 4",
     "codeLines": [
      5,
      7
     ],
     "facts": {
      "jumps": 2,
      "far": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "1",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "i",
        "4": "end"
       }
      }
     ]
    },
    {
     "title": "i = 3",
     "description": "farthest = 4",
     "codeLines": [
      5
     ],
     "facts": {
      "jumps": 2,
      "far": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "3",
        "1",
        "1",
        "4"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "i",
        "4": "end"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "2 jumps.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "2"
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
