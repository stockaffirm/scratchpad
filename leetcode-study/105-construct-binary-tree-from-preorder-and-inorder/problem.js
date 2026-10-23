window.PROBLEM = {
 "id": "105",
 "title": "Construct Binary Tree from Preorder and Inorder Traversal",
 "level": "Medium",
 "topic": "Binary Tree - General",
 "question": "Rebuild a tree from its preorder list (root first) and inorder list (left, root, right).",
 "why": "preorder [3,9,20,15,7], inorder [9,3,15,20,7] → root 3, left 9, right 20 with children 15 and 7.",
 "idea": "The next preorder value is always a root. Find it in the inorder list: everything to its left is the left subtree.",
 "input": "preorder=[3,9,20,15,7], inorder=[9,3,15,20,7]",
 "output": "[3,9,20,null,null,15,7]",
 "notes": [],
 "best": "The next preorder value is always a root. Find it in the inorder list: everything to its left is the left subtree.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def buildTree(self, preorder, inorder):\n        pos = {v: i for i, v in enumerate(inorder)}\n        it = iter(preorder)\n        def build(lo, hi):\n            if lo > hi: return None\n            v = next(it); node = TreeNode(v)\n            m = pos[v]\n            node.left = build(lo, m - 1); node.right = build(m + 1, hi)\n            return node\n        return build(0, len(inorder) - 1)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "The next preorder value is always a root. Find it in the inorder list: everything to its left is the left subtree.",
   "frames": [
    {
     "title": "Inputs",
     "description": "Preorder starts with the root.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "preorder",
       "cells": [
        "3",
        "9",
        "20",
        "15",
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "inorder",
       "cells": [
        "9",
        "3",
        "15",
        "20",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Root = 3",
     "description": "In the inorder list, 9 is left of 3 and 15,20,7 are right of it.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "preorder",
       "cells": [
        "3",
        "9",
        "20",
        "15",
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "inorder",
       "cells": [
        "9",
        "3",
        "15",
        "20",
        "7"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Left: 9",
     "description": "Next preorder value 9 — left subtree [9] is a single node.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "preorder",
       "cells": [
        "3",
        "9",
        "20",
        "15",
        "7"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "inorder",
       "cells": [
        "9",
        "3",
        "15",
        "20",
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Right: 20",
     "description": "Next is 20; in inorder, 15 is left of 20 and 7 is right.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "preorder",
       "cells": [
        "3",
        "9",
        "20",
        "15",
        "7"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "inorder",
       "cells": [
        "9",
        "3",
        "15",
        "20",
        "7"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leaves 15, 7",
     "description": "Remaining preorder values become 20's children.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "9",
        "20"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "·",
        "15",
        "7"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
