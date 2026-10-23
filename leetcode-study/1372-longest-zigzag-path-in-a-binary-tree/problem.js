window.PROBLEM = {
 "id": "1372",
 "title": "Longest ZigZag Path in a Binary Tree",
 "level": "Medium",
 "topic": "Binary Tree - DFS",
 "question": "Longest path that alternates left, right, left, right…",
 "input": "1 → R 2 → L 3 → R 4 (and 2 → R 5)",
 "output": "3",
 "notes": [
  "length counts edges in the zigzag ending here.",
  "Same direction twice restarts at 1."
 ],
 "best": "DFS tracking last direction and current zigzag length.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestZigZag(self, root):\n        best = 0\n        def dfs(node, came_from, length):\n            nonlocal best\n            if not node: return\n            best = max(best, length)\n            dfs(node.left, 'L', length + 1 if came_from == 'R' else 1)\n            dfs(node.right, 'R', length + 1 if came_from == 'L' else 1)\n        dfs(root, '', 0)\n        return best",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Switching direction extends; same direction resets to 1.",
   "frames": [
    {
     "title": "Tree",
     "description": "1 has only a right child 2; 2 has left 3 and right 5; 3 has right 4.",
     "codeLines": [
      4
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
        "·",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "·",
        "3",
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Root",
     "description": "length 0.",
     "codeLines": [
      7
     ],
     "facts": {
      "length": 0,
      "best": 0
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
        "·",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "·",
        "3",
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Go right to 2",
     "description": "Came from R → length 1.",
     "codeLines": [
      9
     ],
     "facts": {
      "length": 1,
      "best": 1
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
        "·",
        "2"
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
        "3",
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Go left to 3",
     "description": "Switched R→L → length 2.",
     "codeLines": [
      8
     ],
     "facts": {
      "length": 2,
      "best": 2
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
        "·",
        "2"
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
        "3",
        "5"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Go right to 4",
     "description": "Switched L→R → length 3.",
     "codeLines": [
      9
     ],
     "facts": {
      "length": 3,
      "best": 3
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
        "·",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "·",
        "·",
        "3",
        "5"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Back at 2 → right to 5",
     "description": "R then R again → reset to 1.",
     "codeLines": [
      9
     ],
     "facts": {
      "length": 1,
      "best": 3
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
        "·",
        "2"
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
        "3",
        "5"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Longest zigzag = 3.",
     "codeLines": [
      11
     ],
     "facts": {
      "best": 3
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
        "·",
        "2"
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
        "3",
        "5"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "·",
        "·",
        "·",
        "·",
        "·",
        "4",
        "·",
        "·"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
