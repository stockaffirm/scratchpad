window.PROBLEM = {
 "id": "124",
 "title": "Binary Tree Maximum Path Sum",
 "level": "Hard",
 "topic": "Binary Tree - General",
 "question": "A path can start and end at any node and moves along edges (it may turn once at a top node). What's the biggest sum of values on a path?",
 "why": "[-10,9,20,null,null,15,7]: the path 15 → 20 → 7 sums to 42.",
 "idea": "For each node, find the best downward 'arm' from each child (ignore negatives). The best path through the node = value + left arm + right arm.",
 "input": "[-10,9,20,null,null,15,7]",
 "output": "42",
 "notes": [],
 "best": "For each node, find the best downward 'arm' from each child (ignore negatives). The best path through the node = value + left arm + right arm.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxPathSum(self, root):\n        best = float('-inf')\n        def gain(n):\n            nonlocal best\n            if not n: return 0\n            l, r = max(gain(n.left), 0), max(gain(n.right), 0)\n            best = max(best, n.val + l + r)\n            return n.val + max(l, r)\n        gain(root)\n        return best",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "For each node, find the best downward 'arm' from each child (ignore negatives). The best path through the node = value + left arm + right arm.",
   "frames": [
    {
     "title": "Tree",
     "description": "Compute each node's best downward arm.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "-10"
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
     "title": "Leaves 15, 7",
     "description": "Arms 15 and 7.",
     "codeLines": [
      9
     ],
     "facts": {
      "best": 15
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "-10"
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
     "description": "Path through 20: 15 + 20 + 7 = 42 → best. Arm up = 20 + 15 = 35.",
     "codeLines": [
      8
     ],
     "facts": {
      "best": 42,
      "arm up": 35
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "-10"
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
       "hot": [
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Node 9",
     "description": "Arm 9; path 9 alone = 9.",
     "codeLines": [
      8
     ],
     "facts": {
      "best": 42
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "-10"
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
     "title": "Root −10",
     "description": "−10 + 9 + 35 = 34 < 42 → best stays 42.",
     "codeLines": [
      8
     ],
     "facts": {
      "best": 42
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "-10"
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
     "title": "Return",
     "description": "42",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "42"
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
