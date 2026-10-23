window.PROBLEM = {
 "id": "547",
 "title": "Number of Provinces",
 "level": "Medium",
 "topic": "Graphs - DFS",
 "question": "Cities are connected directly or indirectly. Count the groups.",
 "input": "[[1,1,0],[1,1,0],[0,0,1]]",
 "output": "2",
 "notes": [
  "Provinces = connected components.",
  "Matrix is symmetric."
 ],
 "best": "DFS flood-fill; count how many times you start one.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findCircleNum(self, isConnected):\n        n, seen, count = len(isConnected), set(), 0\n        def dfs(i):\n            seen.add(i)\n            for j in range(n):\n                if isConnected[i][j] and j not in seen: dfs(j)\n        for i in range(n):\n            if i not in seen:\n                dfs(i); count += 1\n        return count",
   "time": "O(n²)",
   "space": "O(n)",
   "explain": "Each unvisited city starts a new province.",
   "frames": [
    {
     "title": "Matrix",
     "description": "1 means directly connected.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "isConnected",
       "cells": [
        [
         "1",
         "1",
         "0"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "1"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Start at city 0",
     "description": "Not seen → province 1. DFS reaches 1 too.",
     "codeLines": [
      9,
      4
     ],
     "facts": {
      "count": 1
     },
     "rows": [
      {
       "name": "isConnected",
       "cells": [
        [
         "1",
         "1",
         "0"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "1"
        ]
       ],
       "hot": [
        "0,0",
        "0,1"
       ],
       "marks": {},
       "kind": "grid"
      },
      {
       "name": "seen",
       "cells": [
        "0",
        "1"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 1",
     "description": "Already seen → skip.",
     "codeLines": [
      9
     ],
     "facts": {
      "count": 1
     },
     "rows": [
      {
       "name": "seen",
       "cells": [
        "0",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 2",
     "description": "Not seen → province 2.",
     "codeLines": [
      4
     ],
     "facts": {
      "count": 2
     },
     "rows": [
      {
       "name": "isConnected",
       "cells": [
        [
         "1",
         "1",
         "0"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
         "0",
         "0",
         "1"
        ]
       ],
       "hot": [
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      },
      {
       "name": "seen",
       "cells": [
        "0",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "2 provinces: {0,1} and {2}.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "provinces",
       "cells": [
        "{0,1}",
        "{2}"
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
