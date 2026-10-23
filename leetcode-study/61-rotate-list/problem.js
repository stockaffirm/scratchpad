window.PROBLEM = {
 "id": "61",
 "title": "Rotate List",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Rotate the linked list to the right by k places.",
 "why": "1→2→3→4→5, k = 2 → 4→5→1→2→3.",
 "idea": "Make the list a circle, then cut it at the right spot.",
 "input": "1→2→3→4→5, k=2",
 "output": "4→5→1→2→3",
 "notes": [],
 "best": "Make the list a circle, then cut it at the right spot.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def rotateRight(self, head, k):\n        if not head or not head.next: return head\n        n, tail = 1, head\n        while tail.next: tail = tail.next; n += 1\n        k %= n\n        if k == 0: return head\n        new_tail = head\n        for _ in range(n - k - 1): new_tail = new_tail.next\n        new_head = new_tail.next\n        new_tail.next = None; tail.next = head\n        return new_head",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Make the list a circle, then cut it at the right spot.",
   "frames": [
    {
     "title": "Length",
     "description": "n = 5, k = 2 → new head is node 4 (n − k = 3 nodes stay in front).",
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
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "tail"
       },
       "kind": "list"
      }
     ]
    },
    {
     "title": "Close the loop",
     "description": "tail.next = head.",
     "codeLines": [
      11
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
        "5",
        "↺"
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
     "title": "Cut",
     "description": "Break after node 3; node 4 becomes the head.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "4",
        "5",
        "1",
        "2",
        "3"
       ],
       "hot": [
        0
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
