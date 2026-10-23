window.PROBLEM = {
 "id": "2300",
 "title": "Successful Pairs of Spells and Potions",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "A pair succeeds if spell × potion ≥ success. For each spell, count the potions that work.",
 "input": "spells=[5,1,3], potions=[1,2,3,4,5], success=7",
 "output": "[4, 0, 3]",
 "notes": [
  "need = ceil(success / s) via (success + s - 1) // s.",
  "Everything right of the cut-off works."
 ],
 "best": "Sort potions once; bisect for the cut-off per spell.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def successfulPairs(self, spells, potions, success):\n        potions.sort()\n        m = len(potions)\n        return [m - bisect_left(potions, (success + s - 1) // s) for s in spells]",
   "time": "O((n+m) log m)",
   "space": "O(1)",
   "explain": "Binary search the smallest potion that is strong enough.",
   "frames": [
    {
     "title": "Sorted potions",
     "description": "For each spell, find the first potion ≥ need.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "potions",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Spell 5",
     "description": "need = ceil(7/5) = 2 → first at index 1 → 5 − 1 = 4 pairs.",
     "codeLines": [
      5
     ],
     "facts": {
      "count": 4
     },
     "rows": [
      {
       "name": "potions",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [
        1,
        2,
        3,
        4
       ],
       "marks": {
        "1": "need"
       }
      }
     ]
    },
    {
     "title": "Spell 1",
     "description": "need = 7 → none (index 5) → 0 pairs.",
     "codeLines": [
      5
     ],
     "facts": {
      "count": 0
     },
     "rows": [
      {
       "name": "potions",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Spell 3",
     "description": "need = ceil(7/3) = 3 → index 2 → 5 − 2 = 3 pairs.",
     "codeLines": [
      5
     ],
     "facts": {
      "count": 3
     },
     "rows": [
      {
       "name": "potions",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [
        2,
        3,
        4
       ],
       "marks": {
        "2": "need"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "[4, 0, 3]",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "4",
        "0",
        "3"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
