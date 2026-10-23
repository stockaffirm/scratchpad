window.PROBLEM = {
 "id": "92",
 "title": "Reverse Linked List II",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Reverse only the part of the list from position left to position right.",
 "why": "1→2→3→4→5, left 2, right 4 → 1→4→3→2→5.",
 "idea": "Walk to the node before the section, then repeatedly move the next node to the front of the section.",
 "input": "1→2→3→4→5, left=2, right=4",
 "output": "1→4→3→2→5",
 "notes": [],
 "best": "Walk to the node before the section, then repeatedly move the next node to the front of the section.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def reverseBetween(self, head, left, right):\n        dummy = ListNode(0, head); prev = dummy\n        for _ in range(left - 1): prev = prev.next\n        cur = prev.next\n        for _ in range(right - left):\n            nxt = cur.next\n            cur.next = nxt.next\n            nxt.next = prev.next\n            prev.next = nxt\n        return dummy.next",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Walk to the node before the section, then repeatedly move the next node to the front of the section.",
   "frames": [
    {
     "title": "Start",
     "description": "prev stops at 1 (just before position 2); cur = 2.",
     "codeLines": [
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
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
        3
       ],
       "marks": {
        "0": "prev",
        "1": "cur"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Move 3 to the front",
     "description": "nxt = 3 jumps in front of 2 (right after prev).",
     "codeLines": [
      7,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "3",
        "2",
        "4",
        "5"
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
     "title": "Move 4 to the front",
     "description": "nxt = 4 jumps in front of 3.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "4",
        "3",
        "2",
        "5"
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
     "title": "Return",
     "description": "Section 2–4 is reversed.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "4",
        "3",
        "2",
        "5"
       ],
       "hot": [
        1,
        2,
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    }
   ]
  }
 ]
};
