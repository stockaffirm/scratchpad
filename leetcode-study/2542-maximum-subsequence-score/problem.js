window.PROBLEM = {
 "id": "2542",
 "title": "Maximum Subsequence Score",
 "level": "Medium",
 "topic": "Heap / Priority Queue",
 "question": "Pick k indices; score = (sum of nums1) × (min of nums2). Maximise it.",
 "input": "nums1=[1,3,3,2], nums2=[2,1,3,4], k=3",
 "output": "12",
 "notes": [
  "Descending nums2 → current n2 is the minimum so far.",
  "Drop the smallest nums1 to keep the sum large."
 ],
 "best": "Sort by nums2 desc so the current n2 is the min; a min-heap keeps the k largest nums1.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxScore(self, nums1, nums2, k):\n        pairs = sorted(zip(nums2, nums1), reverse=True)\n        heap, total, best = [], 0, 0\n        for n2, n1 in pairs:\n            heappush(heap, n1); total += n1\n            if len(heap) > k: total -= heappop(heap)\n            if len(heap) == k: best = max(best, total * n2)\n        return best",
   "time": "O(n log n)",
   "space": "O(k)",
   "explain": "Fix the min via sorted nums2; keep the k biggest nums1.",
   "frames": [
    {
     "title": "Sort",
     "description": "Pairs (nums2, nums1) by nums2 descending.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "(n2, n1)",
       "cells": [
        "(4,2)",
        "(3,3)",
        "(2,1)",
        "(1,3)"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "n2 = 4",
     "description": "Add 2.",
     "codeLines": [
      6,
      7,
      8
     ],
     "facts": {
      "total": 2,
      "best": 0
     },
     "rows": [
      {
       "name": "(n2, n1)",
       "cells": [
        "(4,2)",
        "(3,3)",
        "(2,1)",
        "(1,3)"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "heap",
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
     "title": "n2 = 3",
     "description": "Add 3.",
     "codeLines": [
      6,
      7,
      8
     ],
     "facts": {
      "total": 5,
      "best": 0
     },
     "rows": [
      {
       "name": "(n2, n1)",
       "cells": [
        "(4,2)",
        "(3,3)",
        "(2,1)",
        "(1,3)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "heap",
       "cells": [
        "2",
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
     "title": "n2 = 2",
     "description": "Add 1. Score 6×2 = 12.",
     "codeLines": [
      6,
      7,
      8
     ],
     "facts": {
      "total": 6,
      "best": 12
     },
     "rows": [
      {
       "name": "(n2, n1)",
       "cells": [
        "(4,2)",
        "(3,3)",
        "(2,1)",
        "(1,3)"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "heap",
       "cells": [
        "1",
        "2",
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
     "title": "n2 = 1",
     "description": "Add 3. Drop the smallest. Score 8×1 = 8.",
     "codeLines": [
      6,
      7,
      8
     ],
     "facts": {
      "total": 8,
      "best": 12
     },
     "rows": [
      {
       "name": "(n2, n1)",
       "cells": [
        "(4,2)",
        "(3,3)",
        "(2,1)",
        "(1,3)"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "heap",
       "cells": [
        "2",
        "3",
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
     "title": "Return",
     "description": "Best score = 12.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "12"
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
