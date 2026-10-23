window.PROBLEM = {
 "id": "1161",
 "title": "Maximum Level Sum of a Binary Tree",
 "level": "Medium",
 "topic": "Binary Tree - BFS",
 "question": "Which level (1-indexed) has the largest sum?",
 "input": "[1, 7, 0, 7, -8, null, null]",
 "output": "2",
 "notes": [
  "Levels are 1-indexed.",
  "Keep the smallest level on ties (strict >)."
 ],
 "best": "BFS by level, tracking the best sum and its level.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxLevelSum(self, root):\n        q, level, best, ans = deque([root]), 1, -inf, 1\n        while q:\n            s = 0\n            for _ in range(len(q)):\n                n = q.popleft(); s += n.val\n                if n.left: q.append(n.left)\n                if n.right: q.append(n.right)\n            if s > best: best, ans = s, level\n            level += 1\n        return ans",
   "time": "O(n)",
   "space": "O(w)",
   "explain": "Sum each level during BFS.",
   "frames": [
    {
     "title": "Tree",
     "description": "Sum each level.",
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
        "7",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "7",
        "-8",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 1",
     "description": "Sum = 1 → best so far.",
     "codeLines": [
      7,
      10
     ],
     "facts": {
      "sum": 1,
      "best": 1,
      "level": 1
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
        "7",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "7",
        "-8",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 2",
     "description": "7 + 0 = 7 → new best.",
     "codeLines": [
      10
     ],
     "facts": {
      "sum": 7,
      "best": 7,
      "level": 2
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
        "7",
        "0"
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
        "7",
        "-8",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 3",
     "description": "7 + (-8) = -1 → not better.",
     "codeLines": [
      10
     ],
     "facts": {
      "sum": -1,
      "best": 7,
      "level": 2
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
        "7",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "7",
        "-8",
        "·",
        "·"
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
     "title": "Return",
     "description": "Level 2.",
     "codeLines": [
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "ans",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
