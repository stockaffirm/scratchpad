window.PROBLEM = {
 "id": "23",
 "title": "Merge k Sorted Lists",
 "level": "Hard",
 "topic": "Divide & Conquer",
 "question": "Merge k sorted linked lists into one sorted list.",
 "why": "[1,4,5], [1,3,4], [2,6] → 1,1,2,3,4,4,5,6.",
 "idea": "Keep a min-heap of the front node of each list; always take the smallest.",
 "input": "[[1,4,5],[1,3,4],[2,6]]",
 "output": "[1,1,2,3,4,4,5,6]",
 "notes": [],
 "best": "Keep a min-heap of the front node of each list; always take the smallest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def mergeKLists(self, lists):\n        heap = [(l.val, i, l) for i, l in enumerate(lists) if l]; heapify(heap)\n        dummy = cur = ListNode()\n        while heap:\n            _, i, node = heappop(heap)\n            cur.next = node; cur = node\n            if node.next: heappush(heap, (node.next.val, i, node.next))\n        return dummy.next",
   "time": "O(N log k)",
   "space": "O(k)",
   "explain": "Keep a min-heap of the front node of each list; always take the smallest.",
   "frames": [
    {
     "title": "Fronts",
     "description": "Heap holds the first node of each list: 1, 1, 2.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (fronts)",
       "cells": [
        "1",
        "1",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "merged",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take 1",
     "description": "Pop the smallest, push that list's next node.",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (fronts)",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "merged",
       "cells": [
        "1"
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
     "title": "Take 1",
     "description": "Pop the smallest, push that list's next node.",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (fronts)",
       "cells": [
        "2",
        "3",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "merged",
       "cells": [
        "1",
        "1"
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
     "title": "Take 2",
     "description": "Pop the smallest, push that list's next node.",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (fronts)",
       "cells": [
        "3",
        "4",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "merged",
       "cells": [
        "1",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Take 6",
     "description": "Pop the smallest, push that list's next node.",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (fronts)",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "merged",
       "cells": [
        "1",
        "1",
        "2",
        "3",
        "4",
        "4",
        "5",
        "6"
       ],
       "hot": [
        7
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Return",
     "description": "Merged list.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "merged",
       "cells": [
        "1",
        "1",
        "2",
        "3",
        "4",
        "4",
        "5",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
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
