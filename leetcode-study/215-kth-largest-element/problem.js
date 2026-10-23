window.PROBLEM = {
 "id": "215",
 "title": "Kth Largest Element in an Array",
 "level": "Medium",
 "topic": "Heap / Priority Queue",
 "question": "Find the kth largest value (not kth distinct).",
 "input": "[3, 2, 1, 5, 6, 4], k = 2",
 "output": "5",
 "notes": [
  "Pop the smallest whenever size exceeds k.",
  "Python's heapq is a min-heap."
 ],
 "best": "Keep a min-heap of size k; its smallest is the answer.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findKthLargest(self, nums, k):\n        heap = []\n        for x in nums:\n            heappush(heap, x)\n            if len(heap) > k: heappop(heap)\n        return heap[0]",
   "time": "O(n log k)",
   "space": "O(k)",
   "explain": "Min-heap of the k largest; the root is the kth largest.",
   "frames": [
    {
     "title": "Input",
     "description": "Keep only the 2 largest seen so far.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [],
       "marks": {}
      },
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
     "title": "x = 3",
     "description": "Push 3.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "x"
       }
      },
      {
       "name": "heap (min first)",
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
     "title": "x = 2",
     "description": "Push 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "x"
       }
      },
      {
       "name": "heap (min first)",
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
     "title": "x = 1",
     "description": "Push 1. Too big → pop smallest (1).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "x"
       }
      },
      {
       "name": "heap (min first)",
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
     "title": "x = 5",
     "description": "Push 5. Too big → pop smallest (2).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "x"
       }
      },
      {
       "name": "heap (min first)",
       "cells": [
        "3",
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 6",
     "description": "Push 6. Too big → pop smallest (3).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "x"
       }
      },
      {
       "name": "heap (min first)",
       "cells": [
        "5",
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
     "title": "x = 4",
     "description": "Push 4. Too big → pop smallest (4).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "3",
        "2",
        "1",
        "5",
        "6",
        "4"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "x"
       }
      },
      {
       "name": "heap (min first)",
       "cells": [
        "5",
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
     "title": "Return",
     "description": "heap[0] = 5 is the 2nd largest.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap",
       "cells": [
        "5",
        "6"
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
