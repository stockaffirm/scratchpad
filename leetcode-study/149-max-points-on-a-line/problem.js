window.PROBLEM = {
 "id": "149",
 "title": "Max Points on a Line",
 "level": "Hard",
 "topic": "Math",
 "question": "Given points on a plane, what's the most points that lie on one straight line?",
 "why": "Points (1,1), (2,2), (3,3), (1,4): the first three are on a line → 3.",
 "idea": "From each point, count other points by their slope; the biggest group + the point itself is a line.",
 "input": "[[1,1],[2,2],[3,3],[1,4]]",
 "output": "3",
 "notes": [],
 "best": "From each point, count other points by their slope; the biggest group + the point itself is a line.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxPoints(self, points):\n        best = 1\n        for i, (x1, y1) in enumerate(points):\n            slopes = Counter()\n            for x2, y2 in points[i+1:]:\n                g = gcd(y2 - y1, x2 - x1)\n                slopes[((y2 - y1) // g, (x2 - x1) // g)] += 1\n            if slopes: best = max(best, max(slopes.values()) + 1)\n        return best",
   "time": "O(n²)",
   "space": "O(n)",
   "explain": "From each point, count other points by their slope; the biggest group + the point itself is a line.",
   "frames": [
    {
     "title": "Points",
     "description": "From (1,1), measure the slope to each other point.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "points",
       "cells": [
        "(1,1)",
        "(2,2)",
        "(3,3)",
        "(1,4)"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slopes",
     "description": "(2,2) → (1,1); (3,3) → (1,1); (1,4) → (1,0). Reduce with gcd.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "slope → count",
       "cells": [
        "(1,1): 2",
        "(1,0): 1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Best group",
     "description": "2 points share a slope + the start point = 3.",
     "codeLines": [
      9
     ],
     "facts": {
      "best": 3
     },
     "rows": [
      {
       "name": "on one line",
       "cells": [
        "(1,1)",
        "(2,2)",
        "(3,3)"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "3",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "3"
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
