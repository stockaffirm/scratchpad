window.PROBLEM = {
 "id": "2462",
 "title": "Total Cost to Hire K Workers",
 "level": "Medium",
 "topic": "Heap / Priority Queue",
 "question": "Hire k workers in rounds; each round take the cheapest among the first/last `candidates` (ties → lower index).",
 "input": "[17,12,10,2,7,2,11,20,8], k=3, candidates=4",
 "output": "11",
 "notes": [
  "Ties go to the left heap (lower index).",
  "i and j are the next unseen middle workers."
 ],
 "best": "Two min-heaps for the two ends; refill from the middle after each hire.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def totalCost(self, costs, k, candidates):\n        n = len(costs)\n        left = costs[:candidates]; heapify(left)\n        right = costs[max(candidates, n - candidates):]; heapify(right)\n        i, j, total = candidates, n - candidates - 1, 0\n        for _ in range(k):\n            if not right or (left and left[0] <= right[0]):\n                total += heappop(left)\n                if i <= j: heappush(left, costs[i]); i += 1\n            else:\n                total += heappop(right)\n                if i <= j: heappush(right, costs[j]); j -= 1\n        return total",
   "time": "O((k+c) log c)",
   "space": "O(c)",
   "explain": "Two heaps hold the front and back candidate windows.",
   "frames": [
    {
     "title": "Two heaps",
     "description": "Left: first 4. Right: last 4.",
     "codeLines": [
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "left heap",
       "cells": [
        "2",
        "10",
        "12",
        "17"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "right heap",
       "cells": [
        "2",
        "8",
        "11",
        "20"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Hire 1",
     "description": "Left 2 ≤ right 2 → hire left. Refill with costs[4] = 7.",
     "codeLines": [
      8,
      9
     ],
     "facts": {
      "total": 2
     },
     "rows": [
      {
       "name": "left heap",
       "cells": [
        "7",
        "10",
        "12",
        "17"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "right heap",
       "cells": [
        "2",
        "8",
        "11",
        "20"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Hire 2",
     "description": "Right 2 < left 7 → hire right. Nothing left to refill.",
     "codeLines": [
      12
     ],
     "facts": {
      "total": 4
     },
     "rows": [
      {
       "name": "left heap",
       "cells": [
        "7",
        "10",
        "12",
        "17"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "right heap",
       "cells": [
        "8",
        "11",
        "20"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Hire 3",
     "description": "Left 7 < right 8 → hire left.",
     "codeLines": [
      9
     ],
     "facts": {
      "total": 11
     },
     "rows": [
      {
       "name": "left heap",
       "cells": [
        "10",
        "12",
        "17"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "right heap",
       "cells": [
        "8",
        "11",
        "20"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "2 + 2 + 7 = 11",
     "codeLines": [
      14
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "11"
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
