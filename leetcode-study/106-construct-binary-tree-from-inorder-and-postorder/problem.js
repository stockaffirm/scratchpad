window.PROBLEM = {
 "id": "106",
 "title": "Construct Binary Tree from Inorder and Postorder Traversal",
 "level": "Medium",
 "topic": "Binary Tree - General",
 "question": "Rebuild a tree from its inorder list and postorder list (left, right, root).",
 "why": "inorder [9,3,15,20,7], postorder [9,15,7,20,3] → the same tree as before.",
 "idea": "The LAST postorder value is the root. Build the right subtree first, then the left.",
 "input": "inorder=[9,3,15,20,7], postorder=[9,15,7,20,3]",
 "output": "[3,9,20,null,null,15,7]",
 "notes": [],
 "best": "The LAST postorder value is the root. Build the right subtree first, then the left.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def buildTree(self, inorder, postorder):\n        pos = {v: i for i, v in enumerate(inorder)}\n        def build(lo, hi):\n            if lo > hi: return None\n            v = postorder.pop(); node = TreeNode(v); m = pos[v]\n            node.right = build(m + 1, hi); node.left = build(lo, m - 1)\n            return node\n        return build(0, len(inorder) - 1)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "The LAST postorder value is the root. Build the right subtree first, then the left.",
   "frames": [
    {
     "title": "Inputs",
     "description": "Postorder ends with the root.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
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
      },
      {
       "name": "postorder",
       "cells": [
        "9",
        "15",
        "7",
        "20",
        "3"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Root = 3",
     "description": "Pop 3. Inorder: 9 is left, 15,20,7 are right.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
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
      },
      {
       "name": "postorder",
       "cells": [
        "9",
        "15",
        "7",
        "20"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Right subtree",
     "description": "Pop 20 (root of the right part). Then 7 (its right child), then 15 (left).",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
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
        2,
        3,
        4
       ],
       "marks": {}
      },
      {
       "name": "postorder",
       "cells": [
        "9"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Left subtree",
     "description": "Pop 9 → left child of 3.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
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
     "title": "Done",
     "description": "The same tree.",
     "codeLines": [
      8
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
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
