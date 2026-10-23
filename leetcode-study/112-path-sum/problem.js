window.PROBLEM = {
 "id": "112",
 "title": "Path Sum",
 "level": "Easy",
 "topic": "Binary Tree - General",
 "question": "Is there a root-to-leaf path whose values add up to the target?",
 "why": "Target 22: 5 → 4 → 11 → 2 = 22 → True.",
 "idea": "Subtract each node's value from the target as you go down; at a leaf the remainder must be 0.",
 "input": "[5,4,8,11,null,13,4,7,2], target=22",
 "output": "True",
 "notes": [],
 "best": "Subtract each node's value from the target as you go down; at a leaf the remainder must be 0.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def hasPathSum(self, root, target):\n        if not root: return False\n        if not root.left and not root.right: return root.val == target\n        return self.hasPathSum(root.left, target - root.val) or self.hasPathSum(root.right, target - root.val)",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Subtract each node's value from the target as you go down; at a leaf the remainder must be 0.",
   "frames": [
    {
     "title": "Tree",
     "description": "Target = 22.",
     "codeLines": [
      2
     ],
     "facts": {
      "target": 22
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "11",
        "·",
        "13",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "7",
        "2",
        "·",
        "·",
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
     "title": "At 5",
     "description": "Remaining 22 − 5 = 17.",
     "codeLines": [
      5
     ],
     "facts": {
      "remaining": 17
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "11",
        "·",
        "13",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "7",
        "2",
        "·",
        "·",
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
     "title": "At 4",
     "description": "17 − 4 = 13.",
     "codeLines": [
      5
     ],
     "facts": {
      "remaining": 13
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "8"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "11",
        "·",
        "13",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "7",
        "2",
        "·",
        "·",
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
     "title": "At 11",
     "description": "13 − 11 = 2.",
     "codeLines": [
      5
     ],
     "facts": {
      "remaining": 2
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "11",
        "·",
        "13",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "7",
        "2",
        "·",
        "·",
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
     "title": "Leaf 2",
     "description": "Leaf value 2 == remaining 2 → True.",
     "codeLines": [
      4
     ],
     "facts": {
      "remaining": 2
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "4",
        "8"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "11",
        "·",
        "13",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "7",
        "2",
        "·",
        "·",
        "·",
        "·",
        "·",
        "·"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
