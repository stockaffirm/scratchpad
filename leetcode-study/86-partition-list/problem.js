window.PROBLEM = {
 "id": "86",
 "title": "Partition List",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Rearrange so all nodes smaller than x come first, then the rest, keeping the original order within each group.",
 "why": "1→4→3→2→5→2 with x = 3 → 1→2→2→4→3→5.",
 "idea": "Build two lists (small and big) and join them.",
 "input": "1→4→3→2→5→2, x=3",
 "output": "1→2→2→4→3→5",
 "notes": [],
 "best": "Build two lists (small and big) and join them.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def partition(self, head, x):\n        small = s = ListNode(); big = b = ListNode()\n        while head:\n            if head.val < x: s.next = head; s = s.next\n            else: b.next = head; b = b.next\n            head = head.next\n        b.next = None; s.next = big.next\n        return small.next",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Build two lists (small and big) and join them.",
   "frames": [
    {
     "title": "Start",
     "description": "x = 3.",
     "codeLines": [
      3
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
        "5",
        "2"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "1 < 3",
     "description": "→ small list.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "small",
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
       "name": "big",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "4 ≥ 3",
     "description": "→ big list.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "small",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "big",
       "cells": [
        "4"
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
     "title": "3 ≥ 3",
     "description": "→ big list.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "small",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "big",
       "cells": [
        "4",
        "3"
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
     "title": "2 < 3",
     "description": "→ small list.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "small",
       "cells": [
        "1",
        "2"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "big",
       "cells": [
        "4",
        "3"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "5 ≥ 3",
     "description": "→ big list.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "small",
       "cells": [
        "1",
        "2"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "big",
       "cells": [
        "4",
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
    },
    {
     "title": "2 < 3",
     "description": "→ small list.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "small",
       "cells": [
        "1",
        "2",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "big",
       "cells": [
        "4",
        "3",
        "5"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Join",
     "description": "small followed by big.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "1",
        "2",
        "2",
        "4",
        "3",
        "5"
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
