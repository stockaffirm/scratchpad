window.PROBLEM = {
 "id": "19",
 "title": "Remove Nth Node From End of List",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Remove the nth node counting from the END of the list.",
 "why": "1→2→3→4→5, n = 2 → remove 4 → 1→2→3→5.",
 "idea": "Send fast n steps ahead, then move both until fast is at the end; slow is just before the target.",
 "input": "1→2→3→4→5, n=2",
 "output": "1→2→3→5",
 "notes": [],
 "best": "Send fast n steps ahead, then move both until fast is at the end; slow is just before the target.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def removeNthFromEnd(self, head, n):\n        dummy = ListNode(0, head); fast = slow = dummy\n        for _ in range(n): fast = fast.next\n        while fast.next:\n            fast, slow = fast.next, slow.next\n        slow.next = slow.next.next\n        return dummy.next",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Send fast n steps ahead, then move both until fast is at the end; slow is just before the target.",
   "frames": [
    {
     "title": "Start",
     "description": "dummy node D in front; both pointers at D.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "D",
        "1",
        "2",
        "3",
        "4",
        "5"
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
     "title": "Gap of n",
     "description": "fast moves 2 steps ahead.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "D",
        "1",
        "2",
        "3",
        "4",
        "5"
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
     "title": "Move together",
     "description": "Both advance until fast reaches the last node.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "D",
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [
        3,
        5
       ],
       "marks": {
        "3": "slow",
        "5": "fast"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Unlink",
     "description": "slow.next (4) is skipped.",
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
        "3",
        "5"
       ],
       "hot": [
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
