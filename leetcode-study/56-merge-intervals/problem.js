window.PROBLEM = {
 "id": "56",
 "title": "Merge Intervals",
 "level": "Medium",
 "topic": "Intervals",
 "question": "Merge all overlapping ranges.",
 "why": "[1,3] and [2,6] overlap → [1,6]; [8,10] and [15,18] stand alone.",
 "idea": "Sort by start. If the next one starts before the last one ends, extend the last one.",
 "input": "[[1,3],[2,6],[8,10],[15,18]]",
 "output": "[[1,6],[8,10],[15,18]]",
 "notes": [],
 "best": "Sort by start. If the next one starts before the last one ends, extend the last one.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def merge(self, intervals):\n        intervals.sort()\n        res = [intervals[0]]\n        for s, e in intervals[1:]:\n            if s <= res[-1][1]: res[-1][1] = max(res[-1][1], e)\n            else: res.append([s, e])\n        return res",
   "time": "O(n log n)",
   "space": "O(n)",
   "explain": "Sort by start. If the next one starts before the last one ends, extend the last one.",
   "frames": [
    {
     "title": "Sorted",
     "description": "Already sorted by start.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,3]",
        "[2,6]",
        "[8,10]",
        "[15,18]"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "[1,3]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "[2,6]",
     "description": "2 ≤ 3 → overlap → [1,6].",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,3]",
        "[2,6]",
        "[8,10]",
        "[15,18]"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "[1,6]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "[8,10]",
     "description": "8 > 6 → new interval.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,3]",
        "[2,6]",
        "[8,10]",
        "[15,18]"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "[1,6]",
        "[8,10]"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "[15,18]",
     "description": "15 > 10 → new interval.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,3]",
        "[2,6]",
        "[8,10]",
        "[15,18]"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "[1,6]",
        "[8,10]",
        "[15,18]"
       ],
       "hot": [
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
