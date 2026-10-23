window.PROBLEM = {
 "id": "1448",
 "title": "Count Good Nodes in Binary Tree",
 "level": "Medium",
 "topic": "Binary Tree - DFS",
 "question": "A node is good if no node on the path from the root is larger. Count them.",
 "input": "[3, 1, 4, 3, null, 1, 5]",
 "output": "4",
 "notes": [
  "Equal to the max counts as good (>=).",
  "Root is always good."
 ],
 "best": "DFS passing the max-so-far down the path.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def goodNodes(self, root):\n        def dfs(node, mx):\n            if not node: return 0\n            good = 1 if node.val >= mx else 0\n            mx = max(mx, node.val)\n            return good + dfs(node.left, mx) + dfs(node.right, mx)\n        return dfs(root, root.val)",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Carry the path maximum down.",
   "frames": [
    {
     "title": "Tree",
     "description": "Carry the maximum seen on the path.",
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
        "1",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 3",
     "description": "3 ≥ 3 → good",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "max so far": 3,
      "count": 1
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
        "1",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 1",
     "description": "1 < 3 → not good",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "max so far": 3,
      "count": 1
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
        "1",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 3",
     "description": "3 ≥ 3 → good",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "max so far": 3,
      "count": 2
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
        "1",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 4",
     "description": "4 ≥ 3 → good",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "max so far": 3,
      "count": 3
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
        "1",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 1",
     "description": "1 < 4 → not good",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "max so far": 4,
      "count": 3
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
        "1",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 5",
     "description": "5 ≥ 4 → good",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "max so far": 4,
      "count": 4
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
        "1",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [
        0,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "4 good nodes.",
     "codeLines": [
      8
     ],
     "facts": {
      "count": 4
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
        "1",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "·",
        "1",
        "5"
       ],
       "hot": [
        0,
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
