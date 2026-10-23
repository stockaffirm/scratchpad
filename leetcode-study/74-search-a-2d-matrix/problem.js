window.PROBLEM = {
 "id": "74",
 "title": "Search a 2D Matrix",
 "level": "Medium",
 "topic": "Binary Search",
 "question": "A grid where each row is sorted and each row starts bigger than the previous row ends. Is the target in it?",
 "why": "Target 3 is in the first row → True.",
 "idea": "Pretend the grid is one long sorted list and binary search it (index → row = i ÷ n, column = i mod n).",
 "input": "[[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=3",
 "output": "True",
 "notes": [],
 "best": "Pretend the grid is one long sorted list and binary search it (index → row = i ÷ n, column = i mod n).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def searchMatrix(self, matrix, target):\n        m, n = len(matrix), len(matrix[0])\n        lo, hi = 0, m * n - 1\n        while lo <= hi:\n            mid = (lo + hi) // 2\n            v = matrix[mid // n][mid % n]\n            if v == target: return True\n            if v < target: lo = mid + 1\n            else: hi = mid - 1\n        return False",
   "time": "O(log mn)",
   "space": "O(1)",
   "explain": "Pretend the grid is one long sorted list and binary search it (index → row = i ÷ n, column = i mod n).",
   "frames": [
    {
     "title": "Grid",
     "description": "Treat it like one list of 12 numbers.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "matrix",
       "cells": [
        [
         "1",
         "3",
         "5",
         "7"
        ],
        [
         "10",
         "11",
         "16",
         "20"
        ],
        [
         "23",
         "30",
         "34",
         "60"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "mid = 5",
     "description": "Value 11 > 3 → search the left part.",
     "codeLines": [
      7,
      10
     ],
     "facts": {
      "lo": 0,
      "hi": 11
     },
     "rows": [
      {
       "name": "matrix",
       "cells": [
        [
         "1",
         "3",
         "5",
         "7"
        ],
        [
         "10",
         "11",
         "16",
         "20"
        ],
        [
         "23",
         "30",
         "34",
         "60"
        ]
       ],
       "hot": [
        "1,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "mid = 2",
     "description": "Value 5 > 3 → left again.",
     "codeLines": [
      10
     ],
     "facts": {
      "lo": 0,
      "hi": 4
     },
     "rows": [
      {
       "name": "matrix",
       "cells": [
        [
         "1",
         "3",
         "5",
         "7"
        ],
        [
         "10",
         "11",
         "16",
         "20"
        ],
        [
         "23",
         "30",
         "34",
         "60"
        ]
       ],
       "hot": [
        "0,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "mid = 0",
     "description": "Value 1 < 3 → right.",
     "codeLines": [
      9
     ],
     "facts": {
      "lo": 0,
      "hi": 1
     },
     "rows": [
      {
       "name": "matrix",
       "cells": [
        [
         "1",
         "3",
         "5",
         "7"
        ],
        [
         "10",
         "11",
         "16",
         "20"
        ],
        [
         "23",
         "30",
         "34",
         "60"
        ]
       ],
       "hot": [
        "0,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "mid = 1",
     "description": "Value 3 = target → True.",
     "codeLines": [
      8
     ],
     "facts": {
      "lo": 1,
      "hi": 1
     },
     "rows": [
      {
       "name": "matrix",
       "cells": [
        [
         "1",
         "3",
         "5",
         "7"
        ],
        [
         "10",
         "11",
         "16",
         "20"
        ],
        [
         "23",
         "30",
         "34",
         "60"
        ]
       ],
       "hot": [
        "0,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    }
   ]
  }
 ]
};
