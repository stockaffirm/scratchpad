window.PROBLEM = {
 "id": "230",
 "title": "Kth Smallest Element in a BST",
 "level": "Medium",
 "topic": "Binary Search Tree",
 "question": "Find the kth smallest value in a BST.",
 "why": "Tree [5,3,6,2,4,null,null,1], k = 3 → inorder is 1,2,3,4,5,6 → the 3rd is 3.",
 "idea": "Inorder gives sorted order; stop at the kth node visited.",
 "input": "[5,3,6,2,4,null,null,1], k=3",
 "output": "3",
 "notes": [],
 "best": "Inorder gives sorted order; stop at the kth node visited.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def kthSmallest(self, root, k):\n        st, n = [], root\n        while st or n:\n            while n: st.append(n); n = n.left\n            n = st.pop(); k -= 1\n            if k == 0: return n.val\n            n = n.right",
   "time": "O(h + k)",
   "space": "O(h)",
   "explain": "Inorder gives sorted order; stop at the kth node visited.",
   "frames": [
    {
     "title": "Tree",
     "description": "Walk inorder (smallest first).",
     "codeLines": [
      3
     ],
     "facts": {
      "k": 3
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
        "3",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "1",
        "·",
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
     "title": "Go far left",
     "description": "Push 5, 3, 2, 1 onto the stack.",
     "codeLines": [
      5
     ],
     "facts": {
      "k": 3
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
        "3",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "1",
        "·",
        "·",
        "·",
        "·",
        "·",
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
     "title": "Visit 1",
     "description": "1st smallest → k = 2.",
     "codeLines": [
      6
     ],
     "facts": {
      "k": 2
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
        "3",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "1",
        "·",
        "·",
        "·",
        "·",
        "·",
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
     "title": "Visit 2",
     "description": "2nd smallest → k = 1.",
     "codeLines": [
      6
     ],
     "facts": {
      "k": 1
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
        "3",
        "6"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "·"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "1",
        "·",
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
     "title": "Visit 3",
     "description": "3rd smallest → k = 0 → return 3.",
     "codeLines": [
      7
     ],
     "facts": {
      "k": 0
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
        "3",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "2",
        "4",
        "·",
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 3",
       "cells": [
        "1",
        "·",
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
    }
   ]
  }
 ]
};
