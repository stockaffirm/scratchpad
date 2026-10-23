window.PROBLEM = {
 "id": "700",
 "title": "Search in a Binary Search Tree",
 "level": "Easy",
 "topic": "Binary Search Tree",
 "question": "Return the subtree rooted at the node whose value equals val.",
 "input": "[4, 2, 7, 1, 3], val = 2",
 "output": "[2, 1, 3]",
 "notes": [
  "BST: left < node < right.",
  "h = height; O(log n) when balanced."
 ],
 "best": "Walk down comparing; no recursion needed.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def searchBST(self, root, val):\n        while root and root.val != val:\n            root = root.left if val < root.val else root.right\n        return root",
   "time": "O(h)",
   "space": "O(1)",
   "explain": "Smaller → left, larger → right.",
   "frames": [
    {
     "title": "Tree",
     "description": "Looking for 2.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "7"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "1",
        "3",
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
     "description": "2 < 4 → go left.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "4"
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
        "7"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "1",
        "3",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "At 2",
     "description": "Found it.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "1",
        "3",
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
     "description": "The subtree 2 → (1, 3).",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "1",
        "3",
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
    }
   ]
  }
 ]
};
