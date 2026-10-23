window.PROBLEM = {
 "id": "1143",
 "title": "Longest Common Subsequence",
 "level": "Medium",
 "topic": "DP - Multidimensional",
 "question": "Length of the longest subsequence common to both strings.",
 "input": "\"abcde\", \"ace\"",
 "output": "3",
 "notes": [
  "Row/col 0 mean the empty prefix → 0.",
  "Match: diagonal + 1. Else: max(up, left)."
 ],
 "best": "2-D DP: match extends the diagonal, otherwise take the better of up/left.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestCommonSubsequence(self, a, b):\n        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\n        for i in range(1, len(a) + 1):\n            for j in range(1, len(b) + 1):\n                if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1\n                else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n        return dp[-1][-1]",
   "time": "O(n·m)",
   "space": "O(n·m)",
   "explain": "dp[i][j] = LCS of a[:i] and b[:j].",
   "frames": [
    {
     "title": "Table",
     "description": "Rows = prefixes of 'abcde', columns = prefixes of 'ace'. Edges are 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 'a'",
     "description": "Match at 1 cell: diagonal + 1.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
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
     "title": "Row 'b'",
     "description": "No match: take the larger of up / left.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 'c'",
     "description": "Match at 1 cell: diagonal + 1.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "3,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 'd'",
     "description": "No match: take the larger of up / left.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Row 'e'",
     "description": "Match at 1 cell: diagonal + 1.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "1",
         "2",
         "3"
        ]
       ],
       "hot": [
        "5,3"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Return",
     "description": "dp[5][3] = 3 ('ace').",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "0",
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "1",
         "1"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "1",
         "2",
         "2"
        ],
        [
         "0",
         "1",
         "2",
         "3"
        ]
       ],
       "hot": [
        "5,3"
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
