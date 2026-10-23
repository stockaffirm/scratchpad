window.PROBLEM = {
 "id": "54",
 "title": "Spiral Matrix",
 "level": "Medium",
 "topic": "Matrix",
 "question": "Read a grid in spiral order: across the top, down the right side, back along the bottom, up the left side, then inward.",
 "why": "[[1,2,3],[4,5,6],[7,8,9]] → 1,2,3,6,9,8,7,4,5.",
 "idea": "Peel off the top row, then rotate what's left so the next side becomes the top.",
 "input": "[[1,2,3],[4,5,6],[7,8,9]]",
 "output": "[1,2,3,6,9,8,7,4,5]",
 "notes": [],
 "best": "Peel off the top row, then rotate what's left so the next side becomes the top.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def spiralOrder(self, m):\n        res = []\n        while m:\n            res += m.pop(0)\n            m = [list(r) for r in zip(*m)][::-1]\n        return res",
   "time": "O(n·m)",
   "space": "O(n·m)",
   "explain": "Peel off the top row, then rotate what's left so the next side becomes the top.",
   "frames": [
    {
     "title": "Start",
     "description": "Take the first row.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "1",
         "2",
         "3"
        ],
        [
         "4",
         "5",
         "6"
        ],
        [
         "7",
         "8",
         "9"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "0,2"
       ],
       "marks": {},
       "kind": "grid"
      },
      {
       "name": "res",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Rotate",
     "description": "Left over: [[4,5,6],[7,8,9]] rotated so the right side is now the top.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "6",
         "9"
        ],
        [
         "5",
         "8"
        ],
        [
         "4",
         "7"
        ]
       ],
       "hot": [
        "0,0",
        "0,1"
       ],
       "marks": {},
       "kind": "grid"
      },
      {
       "name": "res",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take 6, 9",
     "description": "Peel the top row.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "5",
         "8"
        ],
        [
         "4",
         "7"
        ]
       ],
       "hot": [
        "0,0",
        "0,1"
       ],
       "marks": {},
       "kind": "grid"
      },
      {
       "name": "res",
       "cells": [
        "1",
        "2",
        "3",
        "6",
        "9"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Rotate, take 8, 7",
     "description": "Rotate again; the top row is 8, 7.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "m",
       "cells": [
        [
         "5",
         "4"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      },
      {
       "name": "res",
       "cells": [
        "1",
        "2",
        "3",
        "6",
        "9",
        "8",
        "7"
       ],
       "hot": [
        5,
        6
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take 4, then 5",
     "description": "The last pieces.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "1",
        "2",
        "3",
        "6",
        "9",
        "8",
        "7",
        "4",
        "5"
       ],
       "hot": [
        7,
        8
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
