window.PROBLEM = {
 "id": "25",
 "title": "Reverse Nodes in k-Group",
 "level": "Hard",
 "topic": "Linked List",
 "question": "Reverse the list k nodes at a time. If fewer than k nodes remain, leave them as they are.",
 "why": "1→2→3→4→5 with k = 2 → 2→1→4→3→5.",
 "idea": "Check that k nodes exist, reverse them, and attach to the result of handling the rest.",
 "input": "1→2→3→4→5, k=2",
 "output": "2→1→4→3→5",
 "notes": [],
 "best": "Check that k nodes exist, reverse them, and attach to the result of handling the rest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def reverseKGroup(self, head, k):\n        node, cnt = head, 0\n        while node and cnt < k: node = node.next; cnt += 1\n        if cnt < k: return head\n        prev, cur = self.reverseKGroup(node, k), head\n        for _ in range(k):\n            cur.next, prev, cur = prev, cur, cur.next\n        return prev",
   "time": "O(n)",
   "space": "O(n/k)",
   "explain": "Check that k nodes exist, reverse them, and attach to the result of handling the rest.",
   "frames": [
    {
     "title": "Start",
     "description": "k = 2: groups (1,2), (3,4), and a leftover 5.",
     "codeLines": [
      2
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
        0,
        1
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Group 1",
     "description": "Reverse 1,2 → 2,1; it attaches to whatever the rest becomes.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "2",
        "1",
        "3",
        "4",
        "5"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Group 2",
     "description": "Reverse 3,4 → 4,3.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "2",
        "1",
        "4",
        "3",
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
     "title": "Leftover",
     "description": "Only one node (5) left, fewer than k → unchanged.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "list",
       "cells": [
        "2",
        "1",
        "4",
        "3",
        "5"
       ],
       "hot": [
        4
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
