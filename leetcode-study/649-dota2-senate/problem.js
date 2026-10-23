window.PROBLEM = {
 "id": "649",
 "title": "Dota2 Senate",
 "level": "Medium",
 "topic": "Queue",
 "question": "Senators ban opponents in turn. Which party wins?",
 "input": "\"RDD\"",
 "output": "\"Dire\"",
 "notes": [
  "Index + n = the senator's slot in the next round.",
  "The queue that empties loses."
 ],
 "best": "Two queues of indices; the smaller index wins and goes to the back (+n).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def predictPartyVictory(self, senate):\n        n = len(senate)\n        R = deque(i for i, c in enumerate(senate) if c == 'R')\n        D = deque(i for i, c in enumerate(senate) if c == 'D')\n        while R and D:\n            r, d = R.popleft(), D.popleft()\n            if r < d: R.append(r + n)\n            else: D.append(d + n)\n        return 'Radiant' if R else 'Dire'",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Earlier index acts first, bans the other, and re-queues for the next round.",
   "frames": [
    {
     "title": "Queues",
     "description": "R at 0; D at 1 and 2.",
     "codeLines": [
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "R",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "D",
       "cells": [
        "1",
        "2"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Round: 0 vs 1",
     "description": "R(0) acts first → bans D(1). R re-queues as 3.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "R",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "D",
       "cells": [
        "2"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Round: 3 vs 2",
     "description": "D(2) acts first → bans R(3). D re-queues as 5.",
     "codeLines": [
      7,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "R",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "D",
       "cells": [
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "R is empty → Dire wins.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "winner",
       "cells": [
        "Dire"
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
