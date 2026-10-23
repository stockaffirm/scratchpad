window.PROBLEM = {
 "id": "637",
 "title": "Average of Levels in Binary Tree",
 "level": "Easy",
 "topic": "Binary Tree - BFS",
 "question": "What is the average of the values on each level of the tree?",
 "why": "[3,9,20,null,null,15,7]: level averages 3, (9+20)/2 = 14.5, (15+7)/2 = 11.",
 "idea": "BFS level by level; average = sum ÷ count.",
 "input": "[3,9,20,null,null,15,7]",
 "output": "[3.0, 14.5, 11.0]",
 "notes": [],
 "best": "BFS level by level; average = sum ÷ count.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def averageOfLevels(self, root):\n        res, q = [], deque([root])\n        while q:\n            n = len(q); total = 0\n            for _ in range(n):\n                node = q.popleft(); total += node.val\n                if node.left: q.append(node.left)\n                if node.right: q.append(node.right)\n            res.append(total / n)\n        return res",
   "time": "O(n)",
   "space": "O(w)",
   "explain": "BFS level by level; average = sum ÷ count.",
   "frames": [
    {
     "title": "Tree",
     "description": "Average each level.",
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
     "description": "3 / 1 = 3.0",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[3.0]"
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
     "description": "(9 + 20) / 2 = 14.5",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[3.0, 14.5]"
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
     "description": "(15 + 7) / 2 = 11.0",
     "codeLines": [
      10
     ],
     "facts": {
      "res": "[3.0, 14.5, 11.0]"
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
