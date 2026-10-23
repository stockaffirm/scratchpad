window.PROBLEM = {
 "id": "104",
 "title": "Maximum Depth of Binary Tree",
 "level": "Easy",
 "topic": "Binary Tree - DFS",
 "question": "How many levels does the tree have (longest root-to-leaf path)?",
 "input": "[3, 9, 20, null, null, 15, 7]",
 "output": "3",
 "notes": [
  "Empty tree has depth 0.",
  "Space is the recursion depth h."
 ],
 "best": "Recursion: a node's depth is one more than its deeper subtree.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxDepth(self, root):\n        if not root: return 0\n        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Depth = 1 + deeper child.",
   "frames": [
    {
     "title": "Tree",
     "description": "Count levels top to bottom.",
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
    },
    {
     "title": "Leaf 9",
     "description": "Both children empty → depth 1.",
     "codeLines": [
      3,
      4
     ],
     "facts": {
      "depth(9)": 1
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
        "9",
        "20"
       ],
       "hot": [
        0
       ],
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
    },
    {
     "title": "Leaves 15 and 7",
     "description": "Each is a leaf → depth 1.",
     "codeLines": [
      4
     ],
     "facts": {
      "depth(15)": 1,
      "depth(7)": 1
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
    },
    {
     "title": "Node 20",
     "description": "1 + max(1, 1) = 2.",
     "codeLines": [
      4
     ],
     "facts": {
      "depth(20)": 2
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
        "9",
        "20"
       ],
       "hot": [
        1
       ],
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
    },
    {
     "title": "Root 3",
     "description": "1 + max(1, 2) = 3.",
     "codeLines": [
      4
     ],
     "facts": {
      "depth(3)": 3
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
