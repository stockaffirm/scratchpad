window.PROBLEM = {
 "id": "141",
 "title": "Linked List Cycle",
 "level": "Easy",
 "topic": "Linked List",
 "question": "Does the linked list loop back on itself?",
 "why": "3→2→0→−4 and −4 points back to 2 → there's a cycle → True.",
 "idea": "A slow pointer (1 step) and a fast pointer (2 steps): in a loop, fast catches slow.",
 "input": "3 → 2 → 0 → -4 → (back to 2)",
 "output": "True",
 "notes": [],
 "best": "A slow pointer (1 step) and a fast pointer (2 steps): in a loop, fast catches slow.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def hasCycle(self, head):\n        slow = fast = head\n        while fast and fast.next:\n            slow, fast = slow.next, fast.next.next\n            if slow is fast: return True\n        return False",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "A slow pointer (1 step) and a fast pointer (2 steps): in a loop, fast catches slow.",
   "frames": [
    {
     "title": "Start",
     "description": "Both at the head. The tail points back to node 2.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "3",
        "2",
        "0",
        "-4"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "slow,fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Step 1",
     "description": "slow → 2, fast → 0.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "3",
        "2",
        "0",
        "-4"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {
        "1": "slow",
        "2": "fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Step 2",
     "description": "slow → 0, fast → 2 (it looped around).",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "3",
        "2",
        "0",
        "-4"
       ],
       "hot": [
        2,
        1
       ],
       "marks": {
        "1": "fast",
        "2": "slow"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Step 3",
     "description": "Both land on -4 → same node!",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "3",
        "2",
        "0",
        "-4"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "slow,fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "Cycle found → True.",
     "codeLines": [
      6
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
