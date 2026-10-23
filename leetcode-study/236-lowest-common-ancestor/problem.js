window.PROBLEM = {
 "id": "236",
 "title": "Lowest Common Ancestor of a Binary Tree",
 "level": "Medium",
 "topic": "Binary Tree - DFS",
 "question": "Find the lowest node that has both p and q as descendants.",
 "input": "p = 6, q = 4",
 "output": "5",
 "notes": [
  "A node can be its own ancestor.",
  "Returning p or q bubbles the find upward."
 ],
 "best": "Post-order: if both sides found something, this node is the LCA.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def lowestCommonAncestor(self, root, p, q):\n        if not root or root == p or root == q: return root\n        left = self.lowestCommonAncestor(root.left, p, q)\n        right = self.lowestCommonAncestor(root.right, p, q)\n        return root if left and right else left or right",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Each call returns what it found below.",
   "frames": [
    {
     "title": "Tree",
     "description": "Find the meeting point of 6 and 4.",
     "codeLines": [
      2
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
        "5",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "6",
        "2",
        "0",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "7",
        "4",
        "·",
        "·",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Found 6",
     "description": "Subtree under 5-left returns 6.",
     "codeLines": [
      3
     ],
     "facts": {
      "returns": 6
     },
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
        "5",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "6",
        "2",
        "0",
        "8"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "7",
        "4",
        "·",
        "·",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Found 4",
     "description": "Under 2, the right child is 4 → 2 returns 4.",
     "codeLines": [
      3
     ],
     "facts": {
      "2 returns": 4
     },
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
        "5",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "6",
        "2",
        "0",
        "8"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "7",
        "4",
        "·",
        "·",
        "·",
        "·"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Node 5",
     "description": "left = 6, right = 4 → both found → return 5.",
     "codeLines": [
      6
     ],
     "facts": {
      "left": 6,
      "right": 4,
      "returns": 5
     },
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
        "5",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "6",
        "2",
        "0",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "7",
        "4",
        "·",
        "·",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Node 3",
     "description": "left = 5, right = None → return left.",
     "codeLines": [
      6
     ],
     "facts": {
      "returns": 5
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "6",
        "2",
        "0",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "7",
        "4",
        "·",
        "·",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Answer",
     "description": "LCA is 5.",
     "codeLines": [
      6
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
        "5",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "6",
        "2",
        "0",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "7",
        "4",
        "·",
        "·",
        "·",
        "·"
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
