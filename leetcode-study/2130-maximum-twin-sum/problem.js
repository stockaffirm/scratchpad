window.PROBLEM = {
 "id": "2130",
 "title": "Maximum Twin Sum of a Linked List",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Node i pairs with node n-1-i (its twin). Find the largest twin sum.",
 "input": "4 → 2 → 2 → 3",
 "output": "7",
 "notes": [
  "Twins: index i and n-1-i.",
  "Only n/2 pairs to check."
 ],
 "best": "Array copy is simplest; reversing the second half gives O(1) space.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def pairSum(self, head):\n        vals = []\n        while head:\n            vals.append(head.val); head = head.next\n        return max(vals[i] + vals[-1 - i] for i in range(len(vals) // 2))",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Copy to an array, then pair ends inward.",
   "frames": [
    {
     "title": "List",
     "description": "Walk the list and copy values.",
     "codeLines": [
      3,
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "vals",
       "cells": [
        "4",
        "2",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Pair 0",
     "description": "4 + 3 = 7",
     "codeLines": [
      6
     ],
     "facts": {
      "sum": 7
     },
     "rows": [
      {
       "name": "vals",
       "cells": [
        "4",
        "2",
        "2",
        "3"
       ],
       "hot": [
        0,
        3
       ],
       "marks": {
        "0": "i",
        "3": "twin"
       }
      }
     ]
    },
    {
     "title": "Pair 1",
     "description": "2 + 2 = 4",
     "codeLines": [
      6
     ],
     "facts": {
      "sum": 4,
      "max": 7
     },
     "rows": [
      {
       "name": "vals",
       "cells": [
        "4",
        "2",
        "2",
        "3"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {
        "1": "i",
        "2": "twin"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Largest twin sum = 7.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "7"
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
