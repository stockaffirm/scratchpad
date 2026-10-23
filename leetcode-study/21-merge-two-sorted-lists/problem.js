window.PROBLEM = {
 "id": "21",
 "title": "Merge Two Sorted Lists",
 "level": "Easy",
 "topic": "Linked List",
 "question": "Join two sorted linked lists into one sorted list.",
 "why": "[1,2,4] and [1,3,4] → 1,1,2,3,4,4.",
 "idea": "Keep a tail pointer; repeatedly attach the smaller front node.",
 "input": "1→2→4 and 1→3→4",
 "output": "1→1→2→3→4→4",
 "notes": [],
 "best": "Keep a tail pointer; repeatedly attach the smaller front node.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def mergeTwoLists(self, a, b):\n        dummy = cur = ListNode()\n        while a and b:\n            if a.val <= b.val: cur.next, a = a, a.next\n            else: cur.next, b = b, b.next\n            cur = cur.next\n        cur.next = a or b\n        return dummy.next",
   "time": "O(n+m)",
   "space": "O(1)",
   "explain": "Keep a tail pointer; repeatedly attach the smaller front node.",
   "frames": [
    {
     "title": "Start",
     "description": "Compare the two fronts.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "b",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
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
     "title": "Take 1 from a",
     "description": "a front 1 ≤ b front 1",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "b",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
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
     "title": "Take 1 from b",
     "description": "b front 1 < a front 2",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "b",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
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
     "title": "Take 2 from a",
     "description": "a front 2 ≤ b front 3",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "b",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
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
     "title": "Take 3 from b",
     "description": "b front 3 < a front 4",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "b",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "merged",
       "cells": [
        "1",
        "1",
        "2",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Take 4 from a",
     "description": "a front 4 ≤ b front 4",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "b",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "merged",
       "cells": [
        "1",
        "1",
        "2",
        "3",
        "4"
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
     "title": "Attach the rest",
     "description": "One list is empty; append the other.",
     "codeLines": [
      8
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
        "4"
       ],
       "hot": [
        5
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
