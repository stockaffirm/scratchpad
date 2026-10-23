window.PROBLEM = {
 "id": "1466",
 "title": "Reorder Routes to Make All Paths Lead to the City Zero",
 "level": "Medium",
 "topic": "Graphs - DFS",
 "question": "Roads are one-way. Flip the fewest so every city can reach city 0.",
 "input": "n=6, [[0,1],[1,3],[2,3],[4,0],[4,5]]",
 "output": "3",
 "notes": [
  "n cities, n-1 roads → a tree.",
  "Edge a→b seen from a pointing outward must flip."
 ],
 "best": "Treat it as an undirected tree; count edges that point away from 0.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minReorder(self, n, connections):\n        adj = defaultdict(list)\n        for a, b in connections:\n            adj[a].append((b, 1)); adj[b].append((a, 0))\n        seen, changes, stack = {0}, 0, [0]\n        while stack:\n            city = stack.pop()\n            for nxt, cost in adj[city]:\n                if nxt not in seen:\n                    seen.add(nxt); changes += cost; stack.append(nxt)\n        return changes",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Walk away from 0; an edge pointing away costs 1 to flip.",
   "frames": [
    {
     "title": "Roads",
     "description": "0→1, 1→3, 2→3, 4→0, 4→5",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "roads",
       "cells": [
        "0→1",
        "1→3",
        "2→3",
        "4→0",
        "4→5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 0",
     "description": "0→1 points away: flip; 0→4 already toward 0",
     "codeLines": [
      8,
      11
     ],
     "facts": {
      "changes": 1
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "1",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 4",
     "description": "4→5 points away: flip",
     "codeLines": [
      8,
      11
     ],
     "facts": {
      "changes": 2
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "4",
        "5"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "1",
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 5",
     "description": "Nothing new.",
     "codeLines": [
      8,
      11
     ],
     "facts": {
      "changes": 2
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "4",
        "5"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 1",
     "description": "1→3 points away: flip",
     "codeLines": [
      8,
      11
     ],
     "facts": {
      "changes": 3
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "4",
        "5",
        "3"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 3",
     "description": "3→2 already toward 3",
     "codeLines": [
      8,
      11
     ],
     "facts": {
      "changes": 3
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "4",
        "5",
        "3",
        "2"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "City 2",
     "description": "Nothing new.",
     "codeLines": [
      8,
      11
     ],
     "facts": {
      "changes": 3
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "4",
        "5",
        "3",
        "2"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "3 flips: 0→1, 1→3, 4→5.",
     "codeLines": [
      12
     ],
     "facts": {
      "changes": 3
     },
     "rows": [
      {
       "name": "flips",
       "cells": [
        "0→1",
        "1→3",
        "4→5"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
