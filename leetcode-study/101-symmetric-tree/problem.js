window.PROBLEM = {
 "id": "101",
 "title": "Symmetric Tree",
 "level": "Easy",
 "topic": "Binary Tree - General",
 "question": "Is the tree a mirror image of itself around its center?",
 "why": "[1,2,2,3,4,4,3]: the left half is the mirror of the right half → True.",
 "idea": "Compare the left subtree with the right subtree, crossing over: outer with outer, inner with inner.",
 "input": "[1,2,2,3,4,4,3]",
 "output": "True",
 "notes": [],
 "best": "Compare the left subtree with the right subtree, crossing over: outer with outer, inner with inner.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isSymmetric(self, root):\n        def mirror(a, b):\n            if not a and not b: return True\n            if not a or not b or a.val != b.val: return False\n            return mirror(a.left, b.right) and mirror(a.right, b.left)\n        return mirror(root.left, root.right)",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Compare the left subtree with the right subtree, crossing over: outer with outer, inner with inner.",
   "frames": [
    {
     "title": "Tree",
     "description": "Fold along the middle.",
     "codeLines": [
      2
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
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "4",
        "4",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Compare 2 and 2",
     "description": "Roots of the two halves match.",
     "codeLines": [
      7
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
        "3",
        "4",
        "4",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Outer pair",
     "description": "3 and 3 match.",
     "codeLines": [
      6
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
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "4",
        "4",
        "3"
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
     "title": "Inner pair",
     "description": "4 and 4 match.",
     "codeLines": [
      6
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
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "4",
        "4",
        "3"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "True",
     "codeLines": [
      6
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
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "4",
        "4",
        "3"
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
