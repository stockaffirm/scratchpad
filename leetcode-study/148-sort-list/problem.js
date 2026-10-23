window.PROBLEM = {
 "id": "148",
 "title": "Sort List",
 "level": "Medium",
 "topic": "Divide & Conquer",
 "question": "Sort a linked list in O(n log n).",
 "why": "4→2→1→3 becomes 1→2→3→4.",
 "idea": "Merge sort: split the list in half, sort each half, merge the two sorted halves.",
 "input": "4 → 2 → 1 → 3",
 "output": "1 → 2 → 3 → 4",
 "notes": [],
 "best": "Merge sort: split the list in half, sort each half, merge the two sorted halves.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def sortList(self, head):\n        if not head or not head.next: return head\n        slow, fast = head, head.next\n        while fast and fast.next: slow, fast = slow.next, fast.next.next\n        mid, slow.next = slow.next, None\n        return self.merge(self.sortList(head), self.sortList(mid))\n    def merge(self, a, b):\n        dummy = cur = ListNode()\n        while a and b:\n            if a.val <= b.val: cur.next, a = a, a.next\n            else: cur.next, b = b, b.next\n            cur = cur.next\n        cur.next = a or b\n        return dummy.next",
   "time": "O(n log n)",
   "space": "O(log n)",
   "explain": "Merge sort: split the list in half, sort each half, merge the two sorted halves.",
   "frames": [
    {
     "title": "Split",
     "description": "Slow/fast finds the middle → [4,2] and [1,3].",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "left",
       "cells": [
        "4",
        "2"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "right",
       "cells": [
        "1",
        "3"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Split again",
     "description": "[4] [2]   [1] [3] — single nodes are sorted.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "pieces",
       "cells": [
        "[4]",
        "[2]",
        "[1]",
        "[3]"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Merge small",
     "description": "[4]+[2] → [2,4]; [1]+[3] → [1,3].",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "left",
       "cells": [
        "2",
        "4"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "right",
       "cells": [
        "1",
        "3"
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
     "title": "Merge big",
     "description": "[2,4] + [1,3] → [1,2,3,4].",
     "codeLines": [
      14
     ],
     "facts": {},
     "rows": [
      {
       "name": "sorted",
       "cells": [
        "1",
        "2",
        "3",
        "4"
       ],
       "hot": [
        0,
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
