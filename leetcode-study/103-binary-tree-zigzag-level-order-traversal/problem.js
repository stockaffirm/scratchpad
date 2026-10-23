window.PROBLEM = {
 "id": "103",
 "title": "Binary Tree Zigzag Level Order Traversal",
 "level": "Medium",
 "topic": "Binary Tree - BFS",
 "question": "Level order again, but alternate direction: left→right, then right→left, then left→right…",
 "why": "[3,9,20,null,null,15,7] → [[3],[20,9],[15,7]].",
 "idea": "Do normal BFS, and reverse every other level.",
 "input": "[3,9,20,null,null,15,7]",
 "output": "[[3],[20,9],[15,7]]",
 "notes": [],
 "best": "Do normal BFS, and reverse every other level.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def zigzagLevelOrder(self, root):\n        res, q, ltr = [], deque([root] if root else []), True\n        while q:\n            level = []\n            for _ in range(len(q)):\n                node = q.popleft(); level.append(node.val)\n                if node.left: q.append(node.left)\n                if node.right: q.append(node.right)\n            res.append(level if ltr else level[::-1]); ltr = not ltr\n        return res",
   "time": "O(n)",
   "space": "O(w)",
   "explain": "Do normal BFS, and reverse every other level.",
   "frames": [
    {
     "title": "Tree",
     "description": "Alternate direction each level.",
     "codeLines": [
      3
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
     "title": "Level 0 →",
     "description": "[3]",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[[3]]"
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
    },
    {
     "title": "Level 1 ←",
     "description": "Read [9,20] but reversed → [20, 9].",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[[3],[20,9]]"
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
        0,
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
     "title": "Level 2 →",
     "description": "[15, 7]",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[[3],[20,9],[15,7]]"
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
    }
   ]
  }
 ]
};
