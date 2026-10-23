window.PROBLEM = {
 "id": "2336",
 "title": "Smallest Number in Infinite Set",
 "level": "Medium",
 "topic": "Heap / Priority Queue",
 "question": "An infinite set {1,2,3,…}. popSmallest removes and returns the smallest; addBack(x) restores x.",
 "input": "pop, pop, addBack(1), pop, pop",
 "output": "1, 2, 1, 3",
 "notes": [
  "Only numbers below cur can be added back.",
  "The set avoids duplicate heap entries."
 ],
 "best": "Counter for 1,2,3… plus a min-heap for numbers added back.",
 "variants": [
  {
   "name": "Solution",
   "code": "class SmallestInfiniteSet:\n    def __init__(self):\n        self.cur, self.heap, self.inHeap = 1, [], set()\n    def popSmallest(self):\n        if self.heap:\n            x = heappop(self.heap); self.inHeap.remove(x); return x\n        self.cur += 1; return self.cur - 1\n    def addBack(self, num):\n        if num < self.cur and num not in self.inHeap:\n            heappush(self.heap, num); self.inHeap.add(num)",
   "time": "O(log n) per op",
   "space": "O(n)",
   "explain": "A counter for the untouched tail plus a heap of returned numbers.",
   "frames": [
    {
     "title": "Init",
     "description": "cur = 1, heap empty.",
     "codeLines": [
      3
     ],
     "facts": {
      "cur": 1
     },
     "rows": [
      {
       "name": "heap",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "popSmallest()",
     "description": "Heap empty → return cur (1), cur → 2.",
     "codeLines": [
      7
     ],
     "facts": {
      "returned": 1,
      "cur": 2
     },
     "rows": [
      {
       "name": "heap",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "popSmallest()",
     "description": "Return 2, cur → 3.",
     "codeLines": [
      7
     ],
     "facts": {
      "returned": 2,
      "cur": 3
     },
     "rows": [
      {
       "name": "heap",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "addBack(1)",
     "description": "1 < cur → push to heap.",
     "codeLines": [
      9,
      10
     ],
     "facts": {
      "cur": 3
     },
     "rows": [
      {
       "name": "heap",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "popSmallest()",
     "description": "Heap not empty → pop 1.",
     "codeLines": [
      6
     ],
     "facts": {
      "returned": 1,
      "cur": 3
     },
     "rows": [
      {
       "name": "heap",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "popSmallest()",
     "description": "Heap empty → return 3.",
     "codeLines": [
      7
     ],
     "facts": {
      "returned": 3,
      "cur": 4
     },
     "rows": [
      {
       "name": "heap",
       "cells": [
        "·"
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
