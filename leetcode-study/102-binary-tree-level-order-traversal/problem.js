window.PROBLEM = {
 "id": "102",
 "title": "Binary Tree Level Order Traversal",
 "level": "Medium",
 "topic": "Binary Tree - BFS",
 "question": "List the tree's values level by level, left to right.",
 "why": "[3,9,20,null,null,15,7] → [[3],[9,20],[15,7]].",
 "idea": "BFS with a queue; handle one level's worth of nodes at a time.",
 "input": "[3,9,20,null,null,15,7]",
 "output": "[[3],[9,20],[15,7]]",
 "notes": [],
 "best": "BFS with a queue; handle one level's worth of nodes at a time.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def levelOrder(self, root):\n        res, q = [], deque([root] if root else [])\n        while q:\n            level = []\n            for _ in range(len(q)):\n                node = q.popleft(); level.append(node.val)\n                if node.left: q.append(node.left)\n                if node.right: q.append(node.right)\n            res.append(level)\n        return res",
   "time": "O(n)",
   "space": "O(w)",
   "explain": "BFS with a queue; handle one level's worth of nodes at a time.",
   "frames": [
    {
     "title": "Tree",
     "description": "BFS queue.",
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
     "title": "Level 0",
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
     "title": "Level 1",
     "description": "[9, 20]",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[[3],[9,20]]"
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
     "title": "Level 2",
     "description": "[15, 7]",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[[3],[9,20],[15,7]]"
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
