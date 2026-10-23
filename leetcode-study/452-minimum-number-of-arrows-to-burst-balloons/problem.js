window.PROBLEM = {
 "id": "452",
 "title": "Minimum Number of Arrows to Burst Balloons",
 "level": "Medium",
 "topic": "Intervals",
 "question": "Balloons span [start, end] on the x-axis. An arrow at x bursts every balloon covering x. Fewest arrows?",
 "input": "[[10,16],[2,8],[1,6],[7,12]]",
 "output": "2",
 "notes": [
  "Arrow at the earliest end covers the most balloons.",
  "Start > end → a new arrow is needed."
 ],
 "best": "Greedy: sort by end, shoot at the end of a new group.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findMinArrowShots(self, points):\n        points.sort(key=lambda p: p[1])\n        arrows, end = 0, float('-inf')\n        for s, e in points:\n            if s > end:\n                arrows += 1; end = e\n        return arrows",
   "time": "O(n log n)",
   "space": "O(1)",
   "explain": "Shoot at the earliest end; it bursts everything overlapping it.",
   "frames": [
    {
     "title": "Sort by end",
     "description": "[1,6] [2,8] [7,12] [10,16]",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "balloons",
       "cells": [
        "[1,6]",
        "[2,8]",
        "[7,12]",
        "[10,16]"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "[1,6]",
     "description": "Start 1 is past the last arrow → new arrow at 6",
     "codeLines": [
      5,
      7
     ],
     "facts": {
      "arrows": 1
     },
     "rows": [
      {
       "name": "balloons",
       "cells": [
        "[1,6]",
        "[2,8]",
        "[7,12]",
        "[10,16]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "arrow at",
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
     "title": "[2,8]",
     "description": "Start 2 ≤ 6 → already burst",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "arrows": 1
     },
     "rows": [
      {
       "name": "balloons",
       "cells": [
        "[1,6]",
        "[2,8]",
        "[7,12]",
        "[10,16]"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "arrow at",
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
     "title": "[7,12]",
     "description": "Start 7 is past the last arrow → new arrow at 12",
     "codeLines": [
      5,
      7
     ],
     "facts": {
      "arrows": 2
     },
     "rows": [
      {
       "name": "balloons",
       "cells": [
        "[1,6]",
        "[2,8]",
        "[7,12]",
        "[10,16]"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "arrow at",
       "cells": [
        "12"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "[10,16]",
     "description": "Start 10 ≤ 12 → already burst",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "arrows": 2
     },
     "rows": [
      {
       "name": "balloons",
       "cells": [
        "[1,6]",
        "[2,8]",
        "[7,12]",
        "[10,16]"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "arrow at",
       "cells": [
        "12"
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
     "description": "2 arrows (at 6 and 12).",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "2"
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
