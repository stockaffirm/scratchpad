window.PROBLEM = {
 "id": "2352",
 "title": "Equal Row and Column Pairs",
 "level": "Medium",
 "topic": "Hash Map / Set",
 "question": "Count (row, column) pairs that contain the same sequence.",
 "input": "[[3,2,1],[1,7,6],[2,7,7]]",
 "output": "1",
 "notes": [
  "Tuples are hashable; lists are not.",
  "zip(*grid) transposes the grid."
 ],
 "best": "Hash rows as tuples; each column is one O(n) lookup.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def equalPairs(self, grid):\n        rows = Counter(tuple(r) for r in grid)\n        cols = zip(*grid)\n        return sum(rows[c] for c in cols)",
   "time": "O(n²)",
   "space": "O(n²)",
   "explain": "Count rows as tuples, then look up each column.",
   "frames": [
    {
     "title": "Grid",
     "description": "Compare every row with every column.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "3",
         "2",
         "1"
        ],
        [
         "1",
         "7",
         "6"
        ],
        [
         "2",
         "7",
         "7"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Hash rows",
     "description": "Rows become tuples with counts.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "3",
         "2",
         "1"
        ],
        [
         "1",
         "7",
         "6"
        ],
        [
         "2",
         "7",
         "7"
        ]
       ],
       "hot": [
        "2,0",
        "2,1",
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Columns",
     "description": "zip(*grid) gives columns: (3,1,2) (2,7,7) (1,6,7).",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "grid",
       "cells": [
        [
         "3",
         "2",
         "1"
        ],
        [
         "1",
         "7",
         "6"
        ],
        [
         "2",
         "7",
         "7"
        ]
       ],
       "hot": [
        "0,1",
        "1,1",
        "2,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Match",
     "description": "Column 1 = (2,7,7) equals row 2 → count 1.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 2",
       "cells": [
        "2",
        "7",
        "7"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "col 1",
       "cells": [
        "2",
        "7",
        "7"
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
     "title": "Return",
     "description": "1 pair.",
     "codeLines": [
      5
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
