window.PROBLEM = {
 "id": "202",
 "title": "Happy Number",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Repeatedly replace a number by the sum of the squares of its digits. If it reaches 1 it's happy; if it loops forever it isn't.",
 "why": "19 → 82 → 68 → 100 → 1 → happy.",
 "idea": "Remember numbers you've seen; seeing one again means a loop.",
 "input": "19",
 "output": "True",
 "notes": [],
 "best": "Remember numbers you've seen; seeing one again means a loop.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isHappy(self, n):\n        seen = set()\n        while n != 1 and n not in seen:\n            seen.add(n)\n            n = sum(int(d) ** 2 for d in str(n))\n        return n == 1",
   "time": "O(log n)",
   "space": "O(log n)",
   "explain": "Remember numbers you've seen; seeing one again means a loop.",
   "frames": [
    {
     "title": "Start",
     "description": "n = 19",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "19"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "19 →",
     "description": "1² + 9² = 82",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "19",
        "82"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "82 →",
     "description": "8² + 2² = 68",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "19",
        "82",
        "68"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "68 →",
     "description": "6² + 8² = 100",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "19",
        "82",
        "68",
        "100"
       ],
       "hot": [
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "100 →",
     "description": "1² + 0² + 0² = 1",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "19",
        "82",
        "68",
        "100",
        "1"
       ],
       "hot": [
        4
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "Reached 1 → happy.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
