window.PROBLEM = {
 "id": "437",
 "title": "Path Sum III",
 "level": "Medium",
 "topic": "Binary Tree - DFS",
 "question": "Count downward paths whose values sum to target.",
 "input": "[10, 5, -3, 3, 2, null, 11], target=8",
 "output": "2",
 "notes": [
  "A path sums to target if prefix[now] - target was seen earlier on the path.",
  "Undo the count when backtracking."
 ],
 "best": "Prefix-sum hash map along the path turns it into O(n).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def pathSum(self, root, target):\n        count = Counter({0: 1})\n        def dfs(node, cur):\n            if not node: return 0\n            cur += node.val\n            res = count[cur - target]\n            count[cur] += 1\n            res += dfs(node.left, cur) + dfs(node.right, cur)\n            count[cur] -= 1\n            return res\n        return dfs(root, 0)",
   "time": "O(n)",
   "space": "O(h)",
   "explain": "Prefix sums on the path + a hash map of counts.",
   "frames": [
    {
     "title": "Tree",
     "description": "target = 8. Track running sums down the path.",
     "codeLines": [
      3
     ],
     "facts": {
      "target": 8
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "2",
        "·",
        "11"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 10",
     "description": "cur = 10; need earlier prefix 2 → none",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "cur": 10,
      "need": 2,
      "total": 0
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "2",
        "·",
        "11"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 5",
     "description": "cur = 15; need earlier prefix 7 → found! +1",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "cur": 15,
      "need": 7,
      "total": 1
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "2",
        "·",
        "11"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 3",
     "description": "cur = 18; need earlier prefix 10 → none",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "cur": 18,
      "need": 10,
      "total": 1
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "2",
        "·",
        "11"
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
     "description": "cur = 17; need earlier prefix 9 → none",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "cur": 17,
      "need": 9,
      "total": 1
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "2",
        "·",
        "11"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit -3",
     "description": "cur = 7; need earlier prefix -1 → none",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "cur": 7,
      "need": -1,
      "total": 1
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
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
        "2",
        "·",
        "11"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 11",
     "description": "cur = 18; need earlier prefix 10 → found! +1",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "cur": 18,
      "need": 10,
      "total": 2
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "3",
        "2",
        "·",
        "11"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Paths: 5→3 and -3→11.",
     "codeLines": [
      12
     ],
     "facts": {
      "total": 2
     },
     "rows": [
      {
       "name": "level 0",
       "cells": [
        "10"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 1",
       "cells": [
        "5",
        "-3"
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
        "2",
        "·",
        "11"
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
