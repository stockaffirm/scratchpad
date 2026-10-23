window.PROBLEM = {
 "id": "226",
 "title": "Invert Binary Tree",
 "level": "Easy",
 "topic": "Binary Tree - General",
 "question": "Flip the tree like a mirror: swap every node's left and right children.",
 "why": "[4,2,7,1,3,6,9] becomes [4,7,2,9,6,3,1].",
 "idea": "Swap the two children, then do the same inside each child.",
 "input": "[4,2,7,1,3,6,9]",
 "output": "[4,7,2,9,6,3,1]",
 "notes": [],
 "best": "Swap the two children, then do the same inside each child.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def invertTree(self, root):\n        if root:\n            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)\n        return root",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Swap the two children, then do the same inside each child.",
   "frames": [
    {
     "title": "Original",
     "description": "Mirror it.",
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
        "6",
        "9"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Swap leaves",
     "description": "Inside 2 swap (1,3); inside 7 swap (6,9).",
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
        "3",
        "1",
        "9",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Swap root's children",
     "description": "2 and 7 trade places.",
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
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "7",
        "2"
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
        "9",
        "6",
        "3",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Fully mirrored.",
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
        "7",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "9",
        "6",
        "3",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
