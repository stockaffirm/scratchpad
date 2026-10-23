window.PROBLEM = {
 "id": "100",
 "title": "Same Tree",
 "level": "Easy",
 "topic": "Binary Tree - General",
 "question": "Are two binary trees identical — same shape and same values?",
 "why": "Both trees are 1 with children 2 and 3 → True.",
 "idea": "Compare the roots, then compare the left subtrees and the right subtrees the same way.",
 "input": "[1,2,3] and [1,2,3]",
 "output": "True",
 "notes": [],
 "best": "Compare the roots, then compare the left subtrees and the right subtrees the same way.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isSameTree(self, p, q):\n        if not p and not q: return True\n        if not p or not q or p.val != q.val: return False\n        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Compare the roots, then compare the left subtrees and the right subtrees the same way.",
   "frames": [
    {
     "title": "Two trees",
     "description": "Compare node by node.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "p",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "q",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Roots",
     "description": "1 = 1 ✓",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "p",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "q",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Left children",
     "description": "2 = 2 ✓",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "p",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "q",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Right children",
     "description": "3 = 3 ✓",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "p",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "q",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Nothing differed → True.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
