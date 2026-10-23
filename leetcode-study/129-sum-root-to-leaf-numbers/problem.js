window.PROBLEM = {
 "id": "129",
 "title": "Sum Root to Leaf Numbers",
 "level": "Medium",
 "topic": "Binary Tree - General",
 "question": "Each root-to-leaf path spells a number (digits top to bottom). Add up all those numbers.",
 "why": "Paths 4→9→5 = 495, 4→9→1 = 491, 4→0 = 40 → 1026.",
 "idea": "Carry the number so far down the tree: cur = cur × 10 + node value.",
 "input": "[4,9,0,5,1]",
 "output": "1026",
 "notes": [],
 "best": "Carry the number so far down the tree: cur = cur × 10 + node value.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def sumNumbers(self, root):\n        def go(n, cur):\n            if not n: return 0\n            cur = cur * 10 + n.val\n            if not n.left and not n.right: return cur\n            return go(n.left, cur) + go(n.right, cur)\n        return go(root, 0)",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Carry the number so far down the tree: cur = cur × 10 + node value.",
   "frames": [
    {
     "title": "Tree",
     "description": "Build numbers going down.",
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
        "9",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "5",
        "1",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leaf 5",
     "description": "4 → 9 → 5 = 495",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "path": 495
     },
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
        "9",
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "5",
        "1",
        "·",
        "·"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leaf 1",
     "description": "4 → 9 → 1 = 491",
     "codeLines": [
      6
     ],
     "facts": {
      "path": 491
     },
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
        "9",
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "5",
        "1",
        "·",
        "·"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leaf 0",
     "description": "4 → 0 = 40",
     "codeLines": [
      6
     ],
     "facts": {
      "path": 40
     },
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
        "9",
        "0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "5",
        "1",
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
     "description": "495 + 491 + 40 = 1026",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1026"
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
