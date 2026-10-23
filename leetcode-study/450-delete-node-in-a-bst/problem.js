window.PROBLEM = {
 "id": "450",
 "title": "Delete Node in a BST",
 "level": "Medium",
 "topic": "Binary Search Tree",
 "question": "Delete the node with the given key and keep it a valid BST.",
 "input": "[5, 3, 6, 2, 4, null, 7], key = 3",
 "output": "[5, 4, 6, 2, null, null, 7]",
 "notes": [
  "Successor = smallest value in the right subtree.",
  "Copy its value, then delete the successor."
 ],
 "best": "Three cases: leaf, one child, two children (use successor).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def deleteNode(self, root, key):\n        if not root: return None\n        if key < root.val: root.left = self.deleteNode(root.left, key)\n        elif key > root.val: root.right = self.deleteNode(root.right, key)\n        else:\n            if not root.left: return root.right\n            if not root.right: return root.left\n            succ = root.right\n            while succ.left: succ = succ.left\n            root.val = succ.val\n            root.right = self.deleteNode(root.right, succ.val)\n        return root",
   "time": "O(h)",
   "space": "O(h)",
   "explain": "Zero/one child: splice. Two children: copy the in-order successor.",
   "frames": [
    {
     "title": "Tree",
     "description": "Delete 3 (it has two children).",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "3",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Find 3",
     "description": "3 < 5 → go left.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "3",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Two children",
     "description": "Both 2 and 4 exist → need the successor.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "3",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "7"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Successor",
     "description": "Smallest in the right subtree: 4.",
     "codeLines": [
      9,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "3",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
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
     "title": "Copy value",
     "description": "Node 3 becomes 4.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Delete old 4",
     "description": "Remove the successor from the right.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "·",
        "·",
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
