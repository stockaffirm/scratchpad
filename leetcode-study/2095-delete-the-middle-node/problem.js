window.PROBLEM = {
 "id": "2095",
 "title": "Delete the Middle Node of a Linked List",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Delete the middle node (index ⌊n/2⌋) and return the head.",
 "input": "1 → 3 → 4 → 7 → 1 → 2 → 6",
 "output": "1 → 3 → 4 → 1 → 2 → 6",
 "notes": [
  "Starting fast at head.next.next makes slow the node before the middle.",
  "Single-node list returns None."
 ],
 "best": "Slow/fast pointers; start fast one step ahead so slow lands before the middle.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def deleteMiddle(self, head):\n        if not head.next: return None\n        slow, fast = head, head.next.next\n        while fast and fast.next:\n            slow, fast = slow.next, fast.next.next\n        slow.next = slow.next.next\n        return head",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "fast moves twice as fast; slow stops just before the middle.",
   "frames": [
    {
     "title": "Start",
     "description": "slow at 0, fast at 2.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "3",
        "4",
        "7",
        "1",
        "2",
        "6"
       ],
       "hot": [
        0,
        2
       ],
       "marks": {
        "0": "slow",
        "2": "fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Advance",
     "description": "slow +1, fast +2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "3",
        "4",
        "7",
        "1",
        "2",
        "6"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "slow",
        "4": "fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Advance",
     "description": "slow +1, fast +2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "3",
        "4",
        "7",
        "1",
        "2",
        "6"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "slow",
        "6": "fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Unlink",
     "description": "slow.next jumps over the middle (7).",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "3",
        "4",
        "7",
        "1",
        "2",
        "6"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {
        "2": "slow",
        "3": "skip"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "Middle removed.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "3",
        "4",
        "1",
        "2",
        "6"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    }
   ]
  }
 ]
};
