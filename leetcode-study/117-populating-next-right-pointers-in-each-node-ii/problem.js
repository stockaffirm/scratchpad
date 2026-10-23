window.PROBLEM = {
 "id": "117",
 "title": "Populating Next Right Pointers in Each Node II",
 "level": "Medium",
 "topic": "Binary Tree - General",
 "question": "Connect each node's 'next' pointer to the node on its right on the same level (or nothing if it's the last).",
 "why": "Tree [1,2,3,4,5,null,7]: 2→3, 4→5→7, others end.",
 "idea": "Go level by level; link neighbours in each level.",
 "input": "[1,2,3,4,5,null,7]",
 "output": "1→∅, 2→3→∅, 4→5→7→∅",
 "notes": [],
 "best": "Go level by level; link neighbours in each level.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def connect(self, root):\n        level = [root] if root else []\n        while level:\n            for a, b in zip(level, level[1:]): a.next = b\n            level = [c for n in level for c in (n.left, n.right) if c]\n        return root",
   "time": "O(n)",
   "space": "O(w)",
   "explain": "Go level by level; link neighbours in each level.",
   "frames": [
    {
     "title": "Tree",
     "description": "Link neighbours on each level.",
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
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "4",
        "5",
        "·",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 0",
     "description": "Just 1 → nothing to link.",
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
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "4",
        "5",
        "·",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 1",
     "description": "2 → 3",
     "codeLines": [
      5
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
        "3"
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
        "4",
        "5",
        "·",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 2",
     "description": "4 → 5 → 7",
     "codeLines": [
      5
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
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "4",
        "5",
        "·",
        "7"
       ],
       "hot": [
        0,
        1,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "All levels linked.",
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
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "4",
        "5",
        "·",
        "7"
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
