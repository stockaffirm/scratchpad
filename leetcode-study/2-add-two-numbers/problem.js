window.PROBLEM = {
 "id": "2",
 "title": "Add Two Numbers",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Two numbers are stored as linked lists with digits in REVERSE order. Add them and return the sum the same way.",
 "why": "2→4→3 is 342, 5→6→4 is 465 → 807, stored as 7→0→8.",
 "idea": "Add digit by digit like on paper, carrying over.",
 "input": "2→4→3 and 5→6→4",
 "output": "7 → 0 → 8",
 "notes": [],
 "best": "Add digit by digit like on paper, carrying over.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def addTwoNumbers(self, l1, l2):\n        dummy = cur = ListNode(); carry = 0\n        while l1 or l2 or carry:\n            s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry\n            carry, d = divmod(s, 10)\n            cur.next = ListNode(d); cur = cur.next\n            l1 = l1.next if l1 else None; l2 = l2.next if l2 else None\n        return dummy.next",
   "time": "O(max(n,m))",
   "space": "O(max(n,m))",
   "explain": "Add digit by digit like on paper, carrying over.",
   "frames": [
    {
     "title": "Start",
     "description": "Add digits from the least significant end.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "l1",
       "cells": [
        "2",
        "4",
        "3"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "l2",
       "cells": [
        "5",
        "6",
        "4"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "sum",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "2 + 5",
     "description": "= 7, carry 0.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "carry": 0
     },
     "rows": [
      {
       "name": "l1",
       "cells": [
        "2",
        "4",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "l2",
       "cells": [
        "5",
        "6",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "sum",
       "cells": [
        "7"
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
     "title": "4 + 6",
     "description": "= 10 → digit 0, carry 1.",
     "codeLines": [
      6
     ],
     "facts": {
      "carry": 1
     },
     "rows": [
      {
       "name": "l1",
       "cells": [
        "2",
        "4",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "l2",
       "cells": [
        "5",
        "6",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "sum",
       "cells": [
        "7",
        "0"
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
     "title": "3 + 4 + carry 1",
     "description": "= 8, carry 0.",
     "codeLines": [
      6
     ],
     "facts": {
      "carry": 0
     },
     "rows": [
      {
       "name": "l1",
       "cells": [
        "2",
        "4",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "l2",
       "cells": [
        "5",
        "6",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "sum",
       "cells": [
        "7",
        "0",
        "8"
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
     "title": "Return",
     "description": "7 → 0 → 8 (807).",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "7",
        "0",
        "8"
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
