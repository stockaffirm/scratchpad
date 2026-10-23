window.PROBLEM = {
 "id": "399",
 "title": "Evaluate Division",
 "level": "Medium",
 "topic": "Graphs - DFS",
 "question": "Given a/b = 2 and b/c = 3, answer queries like a/c and c/a.",
 "input": "a/b=2, b/c=3 ; queries a/c, c/a",
 "output": "[6.0, 0.1667]",
 "notes": [
  "a/b = 2 means edge a→b weight 2, b→a weight 1/2.",
  "Unknown variable → -1."
 ],
 "best": "Graph with weight v and 1/v; DFS multiplies weights along the path.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def calcEquation(self, equations, values, queries):\n        g = defaultdict(dict)\n        for (a, b), v in zip(equations, values):\n            g[a][b] = v; g[b][a] = 1 / v\n        def dfs(x, y, seen):\n            if x not in g or y not in g: return -1.0\n            if x == y: return 1.0\n            seen.add(x)\n            for nb, w in g[x].items():\n                if nb not in seen:\n                    r = dfs(nb, y, seen)\n                    if r != -1.0: return w * r\n            return -1.0\n        return [dfs(a, b, set()) for a, b in queries]",
   "time": "O(Q·(V+E))",
   "space": "O(V+E)",
   "explain": "Edge weights multiply along a path.",
   "frames": [
    {
     "title": "Build graph",
     "description": "a —2→ b, b —3→ c, plus reverse edges 1/2 and 1/3.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "edges",
       "cells": [
        "a→b 2",
        "b→c 3",
        "b→a ½",
        "c→b ⅓"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Query a/c",
     "description": "Path a → b → c.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Multiply",
     "description": "2 × 3 = 6.",
     "codeLines": [
      13
     ],
     "facts": {},
     "rows": [
      {
       "name": "weights",
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
       "name": "a/c",
       "cells": [
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Query c/a",
     "description": "Path c → b → a: ⅓ × ½ = ⅙.",
     "codeLines": [
      13
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "c",
        "b",
        "a"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "c/a",
       "cells": [
        "0.1667"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "[6.0, 0.1667]",
     "codeLines": [
      15
     ],
     "facts": {},
     "rows": [
      {
       "name": "answers",
       "cells": [
        "6.0",
        "0.1667"
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
