window.PROBLEM = {
 "id": "295",
 "title": "Find Median from Data Stream",
 "level": "Hard",
 "topic": "Heap",
 "question": "Numbers arrive one at a time. After each, report the median so far.",
 "why": "Add 1, add 2 → median 1.5. Add 3 → median 2.",
 "idea": "Keep the smaller half in a max-heap and the larger half in a min-heap; the median is at the tops.",
 "input": "addNum(1), addNum(2), findMedian(), addNum(3), findMedian()",
 "output": "1.5, 2.0",
 "notes": [],
 "best": "Keep the smaller half in a max-heap and the larger half in a min-heap; the median is at the tops.",
 "variants": [
  {
   "name": "Solution",
   "code": "class MedianFinder:\n    def __init__(self): self.lo, self.hi = [], []\n    def addNum(self, x):\n        heappush(self.lo, -x); heappush(self.hi, -heappop(self.lo))\n        if len(self.hi) > len(self.lo): heappush(self.lo, -heappop(self.hi))\n    def findMedian(self):\n        if len(self.lo) > len(self.hi): return -self.lo[0]\n        return (-self.lo[0] + self.hi[0]) / 2",
   "time": "O(log n) add, O(1) median",
   "space": "O(n)",
   "explain": "Keep the smaller half in a max-heap and the larger half in a min-heap; the median is at the tops.",
   "frames": [
    {
     "title": "addNum(1)",
     "description": "Small half [1], big half [].",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "small half (max)",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "big half (min)",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "addNum(2)",
     "description": "Small [1], big [2].",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "small half",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "big half",
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
     "title": "findMedian()",
     "description": "Equal sizes → (1 + 2) / 2 = 1.5",
     "codeLines": [
      8
     ],
     "facts": {
      "median": 1.5
     },
     "rows": [
      {
       "name": "small half",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "big half",
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
     "title": "addNum(3)",
     "description": "Small [2,1], big [3].",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "small half",
       "cells": [
        "2",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "big half",
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
     "title": "findMedian()",
     "description": "Small half is bigger → its top = 2.",
     "codeLines": [
      7
     ],
     "facts": {
      "median": 2
     },
     "rows": [
      {
       "name": "small half",
       "cells": [
        "2",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "big half",
       "cells": [
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
