window.PROBLEM = {
 "id": "36",
 "title": "Valid Sudoku",
 "level": "Medium",
 "topic": "Matrix",
 "question": "Is a partly filled 9×9 Sudoku board valid so far? No repeated digit in any row, column or 3×3 box (empty cells ignored).",
 "why": "The sample board has no repeats anywhere → True.",
 "idea": "Walk every cell and remember (row, digit), (column, digit), (box, digit) in a set; a repeat means invalid.",
 "input": "9×9 board",
 "output": "True",
 "notes": [],
 "best": "Walk every cell and remember (row, digit), (column, digit), (box, digit) in a set; a repeat means invalid.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isValidSudoku(self, board):\n        seen = set()\n        for r in range(9):\n            for c in range(9):\n                v = board[r][c]\n                if v == '.': continue\n                keys = [('r', r, v), ('c', c, v), ('b', r // 3, c // 3, v)]\n                if any(k in seen for k in keys): return False\n                seen.update(keys)\n        return True",
   "time": "O(81)",
   "space": "O(81)",
   "explain": "Walk every cell and remember (row, digit), (column, digit), (box, digit) in a set; a repeat means invalid.",
   "frames": [
    {
     "title": "Board",
     "description": "Check rows, columns and 3×3 boxes together.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "5",
         "3",
         ".",
         ".",
         "7",
         ".",
         ".",
         ".",
         "."
        ],
        [
         "6",
         ".",
         ".",
         "1",
         "9",
         "5",
         ".",
         ".",
         "."
        ],
        [
         ".",
         "9",
         "8",
         ".",
         ".",
         ".",
         ".",
         "6",
         "."
        ],
        [
         "8",
         ".",
         ".",
         ".",
         "6",
         ".",
         ".",
         ".",
         "3"
        ],
        [
         "4",
         ".",
         ".",
         "8",
         ".",
         "3",
         ".",
         ".",
         "1"
        ],
        [
         "7",
         ".",
         ".",
         ".",
         "2",
         ".",
         ".",
         ".",
         "6"
        ],
        [
         ".",
         "6",
         ".",
         ".",
         ".",
         ".",
         "2",
         "8",
         "."
        ],
        [
         ".",
         ".",
         ".",
         "4",
         "1",
         "9",
         ".",
         ".",
         "5"
        ],
        [
         ".",
         ".",
         ".",
         ".",
         "8",
         ".",
         ".",
         "7",
         "9"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 0",
     "description": "5, 3, 7 — all different.",
     "codeLines": [
      8,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "5",
         "3",
         ".",
         ".",
         "7",
         ".",
         ".",
         ".",
         "."
        ],
        [
         "6",
         ".",
         ".",
         "1",
         "9",
         "5",
         ".",
         ".",
         "."
        ],
        [
         ".",
         "9",
         "8",
         ".",
         ".",
         ".",
         ".",
         "6",
         "."
        ],
        [
         "8",
         ".",
         ".",
         ".",
         "6",
         ".",
         ".",
         ".",
         "3"
        ],
        [
         "4",
         ".",
         ".",
         "8",
         ".",
         "3",
         ".",
         ".",
         "1"
        ],
        [
         "7",
         ".",
         ".",
         ".",
         "2",
         ".",
         ".",
         ".",
         "6"
        ],
        [
         ".",
         "6",
         ".",
         ".",
         ".",
         ".",
         "2",
         "8",
         "."
        ],
        [
         ".",
         ".",
         ".",
         "4",
         "1",
         "9",
         ".",
         ".",
         "5"
        ],
        [
         ".",
         ".",
         ".",
         ".",
         "8",
         ".",
         ".",
         "7",
         "9"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "0,4"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Column 0",
     "description": "5, 6, 8, 4, 7 — all different.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "5",
         "3",
         ".",
         ".",
         "7",
         ".",
         ".",
         ".",
         "."
        ],
        [
         "6",
         ".",
         ".",
         "1",
         "9",
         "5",
         ".",
         ".",
         "."
        ],
        [
         ".",
         "9",
         "8",
         ".",
         ".",
         ".",
         ".",
         "6",
         "."
        ],
        [
         "8",
         ".",
         ".",
         ".",
         "6",
         ".",
         ".",
         ".",
         "3"
        ],
        [
         "4",
         ".",
         ".",
         "8",
         ".",
         "3",
         ".",
         ".",
         "1"
        ],
        [
         "7",
         ".",
         ".",
         ".",
         "2",
         ".",
         ".",
         ".",
         "6"
        ],
        [
         ".",
         "6",
         ".",
         ".",
         ".",
         ".",
         "2",
         "8",
         "."
        ],
        [
         ".",
         ".",
         ".",
         "4",
         "1",
         "9",
         ".",
         ".",
         "5"
        ],
        [
         ".",
         ".",
         ".",
         ".",
         "8",
         ".",
         ".",
         "7",
         "9"
        ]
       ],
       "hot": [
        "0,0",
        "1,0",
        "3,0",
        "4,0",
        "5,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Top-left box",
     "description": "5, 3, 6, 9, 8 — all different.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "board",
       "cells": [
        [
         "5",
         "3",
         ".",
         ".",
         "7",
         ".",
         ".",
         ".",
         "."
        ],
        [
         "6",
         ".",
         ".",
         "1",
         "9",
         "5",
         ".",
         ".",
         "."
        ],
        [
         ".",
         "9",
         "8",
         ".",
         ".",
         ".",
         ".",
         "6",
         "."
        ],
        [
         "8",
         ".",
         ".",
         ".",
         "6",
         ".",
         ".",
         ".",
         "3"
        ],
        [
         "4",
         ".",
         ".",
         "8",
         ".",
         "3",
         ".",
         ".",
         "1"
        ],
        [
         "7",
         ".",
         ".",
         ".",
         "2",
         ".",
         ".",
         ".",
         "6"
        ],
        [
         ".",
         "6",
         ".",
         ".",
         ".",
         ".",
         "2",
         "8",
         "."
        ],
        [
         ".",
         ".",
         ".",
         "4",
         "1",
         "9",
         ".",
         ".",
         "5"
        ],
        [
         ".",
         ".",
         ".",
         ".",
         "8",
         ".",
         ".",
         "7",
         "9"
        ]
       ],
       "hot": [
        "0,0",
        "0,1",
        "1,0",
        "2,1",
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "No repeat found.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
