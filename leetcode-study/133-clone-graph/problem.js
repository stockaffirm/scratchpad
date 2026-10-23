window.PROBLEM = {
 "id": "133",
 "title": "Clone Graph",
 "level": "Medium",
 "topic": "Graph - General",
 "question": "Make a deep copy of a connected graph (new nodes with the same connections).",
 "why": "A square graph 1–2–3–4–1 → a new square with its own nodes.",
 "idea": "Walk the graph; create each copy once (remember old→new in a dictionary) and connect copies to copies.",
 "input": "1—2, 2—3, 3—4, 4—1",
 "output": "a new graph with the same shape",
 "notes": [],
 "best": "Walk the graph; create each copy once (remember old→new in a dictionary) and connect copies to copies.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def cloneGraph(self, node):\n        if not node: return None\n        copies = {node: Node(node.val)}\n        stack = [node]\n        while stack:\n            cur = stack.pop()\n            for nb in cur.neighbors:\n                if nb not in copies:\n                    copies[nb] = Node(nb.val); stack.append(nb)\n                copies[cur].neighbors.append(copies[nb])\n        return copies[node]",
   "time": "O(V+E)",
   "space": "O(V)",
   "explain": "Walk the graph; create each copy once (remember old→new in a dictionary) and connect copies to copies.",
   "frames": [
    {
     "title": "Graph",
     "description": "Neighbours: 1→[2,4], 2→[1,3], 3→[2,4], 4→[1,3].",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "graph",
       "cells": [
        "1:[2,4]",
        "2:[1,3]",
        "3:[2,4]",
        "4:[1,3]"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Start",
     "description": "Copy node 1.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "copies",
       "cells": [
        "1→1'"
       ],
       "hot": [
        0
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
     "title": "Visit 1",
     "description": "Neighbours 2 and 4 are new → copy them; link 1' to 2', 4'.",
     "codeLines": [
      10,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "copies",
       "cells": [
        "1→1'",
        "2→2'",
        "4→4'"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "2",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Visit 4, 2",
     "description": "Discover 3; link 4'–3' and 2'–3'. 1 is already copied → just link.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "copies",
       "cells": [
        "1→1'",
        "2→2'",
        "4→4'",
        "3→3'"
       ],
       "hot": [
        3
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
     "title": "Return",
     "description": "copies[1] is the new graph.",
     "codeLines": [
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "copy",
       "cells": [
        "1'",
        "2'",
        "3'",
        "4'"
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
