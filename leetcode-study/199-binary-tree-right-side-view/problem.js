window.PROBLEM = {
 "id": "199",
 "title": "Binary Tree Right Side View",
 "level": "Medium",
 "topic": "Binary Tree - BFS",
 "question": "What do you see standing on the right side of the tree?",
 "input": "[1, 2, 3, null, 5, null, 4]",
 "output": "[1, 3, 4]",
 "notes": [
  "size = len(q) freezes the level width.",
  "w = widest level."
 ],
 "best": "BFS by level; record the last node of each level.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def rightSideView(self, root):\n        res, q = [], deque([root])\n        while q:\n            size = len(q)\n            for i in range(size):\n                node = q.popleft()\n                if i == size - 1: res.append(node.val)\n                if node.left: q.append(node.left)\n                if node.right: q.append(node.right)\n        return res",
   "time": "O(n)",
   "space": "O(w)",
   "explain": "Level by level; the last node of each level is visible.",
   "frames": [
    {
     "title": "Tree",
     "description": "BFS one level at a time.",
     "codeLines": [
      3
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
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "5",
        "·",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 0",
     "description": "Only node 1 → visible.",
     "codeLines": [
      8
     ],
     "facts": {
      "queue": "[1]",
      "res": "[1]"
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "5",
        "·",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 1",
     "description": "Nodes 2, 3. The last one (3) is visible.",
     "codeLines": [
      8
     ],
     "facts": {
      "queue": "[2,3]",
      "res": "[1,3]"
     },
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
        "3"
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
        "5",
        "·",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 2",
     "description": "Nodes 5, 4. The last one (4) is visible.",
     "codeLines": [
      8
     ],
     "facts": {
      "queue": "[5,4]",
      "res": "[1,3,4]"
     },
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
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "5",
        "·",
        "4"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "[1, 3, 4]",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "1",
        "3",
        "4"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
