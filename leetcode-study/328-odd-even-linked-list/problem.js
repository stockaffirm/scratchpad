window.PROBLEM = {
 "id": "328",
 "title": "Odd Even Linked List",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Group all odd-position nodes first, then even-position nodes, in place.",
 "input": "1 → 2 → 3 → 4 → 5",
 "output": "1 → 3 → 5 → 2 → 4",
 "notes": [
  "Keep evenHead to reattach later.",
  "Loop while even and even.next exist."
 ],
 "best": "Build the odd and even chains in one pass, then join them.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def oddEvenList(self, head):\n        if not head: return head\n        odd, even, evenHead = head, head.next, head.next\n        while even and even.next:\n            odd.next = even.next; odd = odd.next\n            even.next = odd.next; even = even.next\n        odd.next = evenHead\n        return head",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Weave two chains, then attach even after odd.",
   "frames": [
    {
     "title": "Input",
     "description": "Positions 1,3,5 are odd; 2,4 are even.",
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
        2,
        4
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Two heads",
     "description": "odd = 1, even = evenHead = 2.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "odd chain",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "even chain",
       "cells": [
        "2"
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
     "title": "Round 1",
     "description": "odd takes 3; even takes 4.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "odd chain",
       "cells": [
        "1",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "even chain",
       "cells": [
        "2",
        "4"
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
     "title": "Round 2",
     "description": "odd takes 5; even.next is None → stop.",
     "codeLines": [
      6,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "odd chain",
       "cells": [
        "1",
        "3",
        "5"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "even chain",
       "cells": [
        "2",
        "4"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Connect",
     "description": "odd tail → evenHead.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "1",
        "3",
        "5",
        "2",
        "4"
       ],
       "hot": [
        3,
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
