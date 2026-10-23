window.PROBLEM = {
 "id": "435",
 "title": "Non-overlapping Intervals",
 "level": "Medium",
 "topic": "Intervals",
 "question": "Remove the fewest intervals so the rest don't overlap.",
 "input": "[[1,2],[2,3],[3,4],[1,3]]",
 "output": "1",
 "notes": [
  "Earliest end leaves the most room.",
  "Touching ends (s == end) are fine."
 ],
 "best": "Greedy: sort by end time and keep any interval that starts after the last kept end.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def eraseOverlapIntervals(self, intervals):\n        intervals.sort(key=lambda x: x[1])\n        end, removed = float('-inf'), 0\n        for s, e in intervals:\n            if s >= end: end = e\n            else: removed += 1\n        return removed",
   "time": "O(n log n)",
   "space": "O(1)",
   "explain": "Sort by end; keep what finishes earliest.",
   "frames": [
    {
     "title": "Sort by end",
     "description": "[1,2] [2,3] [1,3] [3,4]",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[2,3]",
        "[1,3]",
        "[3,4]"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "[1,2]",
     "description": "Starts 1 ≥ end → keep, end = 2",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "removed": 0
     },
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[2,3]",
        "[1,3]",
        "[3,4]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "end",
       "cells": [
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
     "title": "[2,3]",
     "description": "Starts 2 ≥ end → keep, end = 3",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "removed": 0
     },
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[2,3]",
        "[1,3]",
        "[3,4]"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "end",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "[1,3]",
     "description": "Starts 1 < end 3 → overlap, remove",
     "codeLines": [
      5,
      7
     ],
     "facts": {
      "removed": 1
     },
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[2,3]",
        "[1,3]",
        "[3,4]"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "end",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "[3,4]",
     "description": "Starts 3 ≥ end → keep, end = 4",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "removed": 1
     },
     "rows": [
      {
       "name": "intervals",
       "cells": [
        "[1,2]",
        "[2,3]",
        "[1,3]",
        "[3,4]"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "end",
       "cells": [
        "4"
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
     "description": "Removed 1 ([1,3]).",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1"
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
