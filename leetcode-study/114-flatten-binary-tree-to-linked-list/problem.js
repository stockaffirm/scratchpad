window.PROBLEM = {
 "id": "114",
 "title": "Flatten Binary Tree to Linked List",
 "level": "Medium",
 "topic": "Binary Tree - General",
 "question": "Turn the tree into a straight 'linked list' using only right pointers, in preorder order.",
 "why": "[1,2,5,3,4,null,6] → 1→2→3→4→5→6.",
 "idea": "For each node with a left child: attach its right subtree to the end of the left subtree, then move the left subtree to the right.",
 "input": "[1,2,5,3,4,null,6]",
 "output": "1→2→3→4→5→6",
 "notes": [],
 "best": "For each node with a left child: attach its right subtree to the end of the left subtree, then move the left subtree to the right.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def flatten(self, root):\n        cur = root\n        while cur:\n            if cur.left:\n                pre = cur.left\n                while pre.right: pre = pre.right\n                pre.right = cur.right; cur.right = cur.left; cur.left = None\n            cur = cur.right",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "For each node with a left child: attach its right subtree to the end of the left subtree, then move the left subtree to the right.",
   "frames": [
    {
     "title": "Tree",
     "description": "Preorder: 1, 2, 3, 4, 5, 6.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "4",
        "·",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "At 1",
     "description": "Rightmost of the left subtree is 4. Hang 5 off it, then move 2 to the right of 1.",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "chain",
       "cells": [
        "1",
        "2"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "rest",
       "cells": [
        "2→3,4",
        "4→5→6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "At 2",
     "description": "Rightmost of 2's left (3) is 3: hang 4 on it and move 3 right.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "chain",
       "cells": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6"
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
    },
    {
     "title": "Done",
     "description": "Every node now only has a right pointer.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "chain",
       "cells": [
        "1",
        "2",
        "3",
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
