window.PROBLEM = {
 "id": "530",
 "title": "Minimum Absolute Difference in BST",
 "level": "Easy",
 "topic": "Binary Search Tree",
 "question": "Find the smallest difference between any two values in a BST.",
 "why": "[4,2,6,1,3]: in sorted order 1,2,3,4,6 the closest neighbours differ by 1.",
 "idea": "An inorder walk of a BST gives sorted values; only neighbours in that order need comparing.",
 "input": "[4,2,6,1,3]",
 "output": "1",
 "notes": [],
 "best": "An inorder walk of a BST gives sorted values; only neighbours in that order need comparing.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def getMinimumDifference(self, root):\n        prev, best = None, float('inf')\n        def inorder(n):\n            nonlocal prev, best\n            if not n: return\n            inorder(n.left)\n            if prev is not None: best = min(best, n.val - prev)\n            prev = n.val\n            inorder(n.right)\n        inorder(root)\n        return best",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "An inorder walk of a BST gives sorted values; only neighbours in that order need comparing.",
   "frames": [
    {
     "title": "Tree",
     "description": "Inorder visits values smallest to largest.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "1",
        "3",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Inorder order",
     "description": "1, 2, 3, 4, 6",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "visited",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Differences",
     "description": "2−1 = 1, 3−2 = 1, 4−3 = 1, 6−4 = 2",
     "codeLines": [
      8
     ],
     "facts": {
      "best": 1
     },
     "rows": [
      {
       "name": "sorted",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "gaps",
       "cells": [
        "1",
        "1",
        "1",
        "2"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "1",
     "codeLines": [
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1"
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
