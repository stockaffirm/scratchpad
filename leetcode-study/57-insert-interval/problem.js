window.PROBLEM = {
 "id": "57",
 "title": "Insert Interval",
 "level": "Medium",
 "topic": "Intervals",
 "question": "A sorted list of non-overlapping ranges plus one new range. Insert it and merge anything that overlaps.",
 "why": "Insert [4,8] into [1,2],[3,5],[6,7],[8,10],[12,16] → it swallows [3,5],[6,7],[8,10] → [3,10].",
 "idea": "Three phases: copy ranges that end before the new one, merge the overlapping ones, copy the rest.",
 "input": "[[1,2],[3,5],[6,7],[8,10],[12,16]], new=[4,8]",
 "output": "[[1,2],[3,10],[12,16]]",
 "notes": [],
 "best": "Three phases: copy ranges that end before the new one, merge the overlapping ones, copy the rest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def insert(self, intervals, new):\n        res, i, n = [], 0, len(intervals)\n        while i < n and intervals[i][1] < new[0]:\n            res.append(intervals[i]); i += 1\n        while i < n and intervals[i][0] <= new[1]:\n            new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]; i += 1\n        res.append(new)\n        res += intervals[i:]\n        return res",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Three phases: copy ranges that end before the new one, merge the overlapping ones, copy the rest.",
   "frames": [
    {
     "title": "Start",
     "description": "New range [4,8].",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[3,5]",
        "[6,7]",
        "[8,10]",
        "[12,16]"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Before",
     "description": "[1,2] ends before 4 → copy as is.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[3,5]",
        "[6,7]",
        "[8,10]",
        "[12,16]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "[1,2]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Merge",
     "description": "[3,5], [6,7], [8,10] all overlap [4,8] → [3,10].",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[3,5]",
        "[6,7]",
        "[8,10]",
        "[12,16]"
       ],
       "hot": [
        1,
        2,
        3
       ],
       "marks": {}
      },
      {
       "name": "new",
       "cells": [
        "[3,10]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "After",
     "description": "Add merged range, then copy [12,16].",
     "codeLines": [
      8,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "[1,2]",
        "[3,10]",
        "[12,16]"
       ],
       "hot": [
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
