window.PROBLEM = {
 "id": "2352",
 "title": "Equal Row and Column Pairs",
 "level": "Medium",
 "topic": "Hash Map / Set",
 "question": "A square grid. A row and a column are a pair if they hold the same numbers in the same order. How many (row, column) pairs are there?",
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
     ],
     "sub": [
      {
       "description": "row 0 = (3,2,1) → store as a tuple, count 1",
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
          "0,0",
          "0,1",
          "0,2"
         ],
         "marks": {},
         "kind": "grid"
        },
        {
         "name": "row counter",
         "cells": [
          "(3,2,1)"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "row 1 = (1,7,6) → store as a tuple, count 1",
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
          "1,0",
          "1,1",
          "1,2"
         ],
         "marks": {},
         "kind": "grid"
        },
        {
         "name": "row counter",
         "cells": [
          "(3,2,1)",
          "(1,7,6)"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "row 2 = (2,7,7) → store as a tuple, count 1",
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
        },
        {
         "name": "row counter",
         "cells": [
          "(3,2,1)",
          "(1,7,6)",
          "(2,7,7)"
         ],
         "hot": [],
         "marks": {}
        }
       ]
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
     ],
     "sub": [
      {
       "description": "column 0 reads down as (3,1,2) → no matching row",
       "codeLines": [
        4,
        5
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
          "0,0",
          "1,0",
          "2,0"
         ],
         "marks": {},
         "kind": "grid"
        },
        {
         "name": "row counter",
         "cells": [
          "(3,2,1)",
          "(1,7,6)",
          "(2,7,7)"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "matches so far",
         "cells": [
          "0"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "column 1 reads down as (2,7,7) → matches a stored row!",
       "codeLines": [
        4,
        5
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
        },
        {
         "name": "row counter",
         "cells": [
          "(3,2,1)",
          "(1,7,6)",
          "(2,7,7)"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "matches so far",
         "cells": [
          "0",
          "1"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "column 2 reads down as (1,6,7) → no matching row",
       "codeLines": [
        4,
        5
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
          "0,2",
          "1,2",
          "2,2"
         ],
         "marks": {},
         "kind": "grid"
        },
        {
         "name": "row counter",
         "cells": [
          "(3,2,1)",
          "(1,7,6)",
          "(2,7,7)"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "matches so far",
         "cells": [
          "0",
          "1",
          "0"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
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
 ],
 "why": "Row 3 is [2,7,7]. Column 2 reads down as 2,7,7. Same → 1 pair.",
 "idea": "Store each row as a tuple in a counter. Look up every column."
};
