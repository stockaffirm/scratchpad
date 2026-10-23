window.PROBLEM = {
 "id": "82",
 "title": "Remove Duplicates from Sorted List II",
 "level": "Medium",
 "topic": "Linked List",
 "question": "In a sorted linked list, delete EVERY value that appears more than once (not just the extras).",
 "why": "1→2→3→3→4→4→5 → 1→2→5.",
 "idea": "Use a previous pointer; when a value repeats, skip all its copies.",
 "input": "1→2→3→3→4→4→5",
 "output": "1→2→5",
 "notes": [],
 "best": "Use a previous pointer; when a value repeats, skip all its copies.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def deleteDuplicates(self, head):\n        dummy = prev = ListNode(0, head); cur = head\n        while cur:\n            if cur.next and cur.val == cur.next.val:\n                while cur.next and cur.val == cur.next.val: cur = cur.next\n                prev.next = cur.next\n            else: prev = prev.next\n            cur = cur.next\n        return dummy.next",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Use a previous pointer; when a value repeats, skip all its copies.",
   "frames": [
    {
     "title": "Start",
     "description": "Walk with prev and cur.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "2",
        "3",
        "3",
        "4",
        "4",
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "cur"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "1, 2",
     "description": "Unique → prev moves along.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "2",
        "3",
        "3",
        "4",
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
     "title": "3, 3",
     "description": "Repeat → skip both 3s.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "2",
        "3",
        "3",
        "4",
        "4",
        "5"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "4, 4",
     "description": "Repeat → skip both 4s.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "2",
        "4",
        "4",
        "5"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "5 is unique and stays.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "1",
        "2",
        "5"
       ],
       "hot": [
        0,
        1,
        2
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
