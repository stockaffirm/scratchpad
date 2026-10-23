window.PROBLEM = {
 "id": "207",
 "title": "Course Schedule",
 "level": "Medium",
 "topic": "Graph - General",
 "question": "Some courses require others first. Can you finish all courses (is there an order with no circular requirement)?",
 "why": "4 courses: 1 and 2 need 0; 3 needs 1 and 2 → take 0, then 1, 2, then 3 → True.",
 "idea": "Keep taking courses that have no remaining requirements (topological sort). If you finish them all, True.",
 "input": "4 courses, prerequisites [[1,0],[2,0],[3,1],[3,2]]",
 "output": "True",
 "notes": [],
 "best": "Keep taking courses that have no remaining requirements (topological sort). If you finish them all, True.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def canFinish(self, n, prereq):\n        adj, indeg = defaultdict(list), [0] * n\n        for a, b in prereq: adj[b].append(a); indeg[a] += 1\n        q = deque(i for i in range(n) if indeg[i] == 0); done = 0\n        while q:\n            c = q.popleft(); done += 1\n            for nxt in adj[c]:\n                indeg[nxt] -= 1\n                if indeg[nxt] == 0: q.append(nxt)\n        return done == n",
   "time": "O(V+E)",
   "space": "O(V+E)",
   "explain": "Keep taking courses that have no remaining requirements (topological sort). If you finish them all, True.",
   "frames": [
    {
     "title": "Count requirements",
     "description": "Number of prerequisites per course: [0,1,1,2].",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "course",
       "cells": [
        "0",
        "1",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "needs",
       "cells": [
        "0",
        "1",
        "1",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take 0",
     "description": "No requirements. Taking it lowers 1 and 2 to 0.",
     "codeLines": [
      7,
      9
     ],
     "facts": {
      "done": 1
     },
     "rows": [
      {
       "name": "course",
       "cells": [
        "0",
        "1",
        "2",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "needs",
       "cells": [
        "0",
        "0",
        "0",
        "2"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "queue",
       "cells": [
        "1",
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take 1, then 2",
     "description": "Course 3 goes 2 → 1 → 0.",
     "codeLines": [
      10
     ],
     "facts": {
      "done": 3
     },
     "rows": [
      {
       "name": "course",
       "cells": [
        "0",
        "1",
        "2",
        "3"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "needs",
       "cells": [
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "queue",
       "cells": [
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take 3",
     "description": "All 4 courses taken.",
     "codeLines": [
      7
     ],
     "facts": {
      "done": 4
     },
     "rows": [
      {
       "name": "course",
       "cells": [
        "0",
        "1",
        "2",
        "3"
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
     "description": "done == n → True.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
