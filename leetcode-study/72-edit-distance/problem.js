window.PROBLEM = {
 "id": "72",
 "title": "Edit Distance",
 "level": "Medium",
 "topic": "DP - Multidimensional",
 "question": "What's the fewest edits (insert, delete or replace one letter) to turn word1 into word2?",
 "input": "\"horse\" → \"ros\"",
 "output": "3",
 "notes": [
  "First row/column: i or j insertions/deletions.",
  "up = delete, left = insert, diagonal = replace."
 ],
 "best": "2-D DP: match is free; else 1 + min(delete, insert, replace).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minDistance(self, a, b):\n        dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]\n        for i in range(1, len(a) + 1):\n            for j in range(1, len(b) + 1):\n                if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]\n                else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n        return dp[-1][-1]",
   "time": "O(n·m)",
   "space": "O(n·m)",
   "explain": "dp[i][j] = edits to turn a[:i] into b[:j].",
   "frames": [
    {
     "title": "Table",
     "description": "Turning '' into 'ros' takes 0–3 inserts; 'horse' into '' takes 0–5 deletes.",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "0",
         "0",
         "0"
        ],
        [
         "2",
         "0",
         "0",
         "0"
        ],
        [
         "3",
         "0",
         "0",
         "0"
        ],
        [
         "4",
         "0",
         "0",
         "0"
        ],
        [
         "5",
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
     "title": "Row 'h'",
     "description": "No match: 1 + min of three neighbours.",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "1",
         "2",
         "3"
        ],
        [
         "2",
         "0",
         "0",
         "0"
        ],
        [
         "3",
         "0",
         "0",
         "0"
        ],
        [
         "4",
         "0",
         "0",
         "0"
        ],
        [
         "5",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ],
     "sub": [
      {
       "description": "'h' ≠ 'r' → 1 + min(up 1, left 1, diag 0) = 1",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "0",
           "0"
          ],
          [
           "2",
           "0",
           "0",
           "0"
          ],
          [
           "3",
           "0",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
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
       "description": "'h' ≠ 'o' → 1 + min(up 2, left 1, diag 1) = 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "0"
          ],
          [
           "2",
           "0",
           "0",
           "0"
          ],
          [
           "3",
           "0",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "1,2"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'h' ≠ 's' → 1 + min(up 3, left 2, diag 2) = 3",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "0",
           "0",
           "0"
          ],
          [
           "3",
           "0",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "1,3"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      }
     ]
    },
    {
     "title": "Row 'o'",
     "description": "Match → copy the diagonal (free).",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "1",
         "2",
         "3"
        ],
        [
         "2",
         "2",
         "1",
         "2"
        ],
        [
         "3",
         "0",
         "0",
         "0"
        ],
        [
         "4",
         "0",
         "0",
         "0"
        ],
        [
         "5",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "2,2"
       ],
       "marks": {},
       "kind": "grid"
      }
     ],
     "sub": [
      {
       "description": "'o' ≠ 'r' → 1 + min(up 1, left 2, diag 1) = 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "0",
           "0"
          ],
          [
           "3",
           "0",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "2,1"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'o' = 'o' → copy diagonal 1 (free)",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "0"
          ],
          [
           "3",
           "0",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "2,2"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'o' ≠ 's' → 1 + min(up 3, left 1, diag 2) = 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "0",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "2,3"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      }
     ]
    },
    {
     "title": "Row 'r'",
     "description": "Match → copy the diagonal (free).",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "1",
         "2",
         "3"
        ],
        [
         "2",
         "2",
         "1",
         "2"
        ],
        [
         "3",
         "2",
         "2",
         "2"
        ],
        [
         "4",
         "0",
         "0",
         "0"
        ],
        [
         "5",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "3,1"
       ],
       "marks": {},
       "kind": "grid"
      }
     ],
     "sub": [
      {
       "description": "'r' = 'r' → copy diagonal 2 (free)",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "0",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "3,1"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'r' ≠ 'o' → 1 + min(up 1, left 2, diag 2) = 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "0"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
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
       "description": "'r' ≠ 's' → 1 + min(up 2, left 2, diag 1) = 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "0",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "3,3"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      }
     ]
    },
    {
     "title": "Row 's'",
     "description": "Match → copy the diagonal (free).",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "1",
         "2",
         "3"
        ],
        [
         "2",
         "2",
         "1",
         "2"
        ],
        [
         "3",
         "2",
         "2",
         "2"
        ],
        [
         "4",
         "3",
         "3",
         "2"
        ],
        [
         "5",
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "4,3"
       ],
       "marks": {},
       "kind": "grid"
      }
     ],
     "sub": [
      {
       "description": "'s' ≠ 'r' → 1 + min(up 2, left 4, diag 3) = 3",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "3",
           "0",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "4,1"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'s' ≠ 'o' → 1 + min(up 2, left 3, diag 2) = 3",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "3",
           "3",
           "0"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "4,2"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'s' = 's' → copy diagonal 2 (free)",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "3",
           "3",
           "2"
          ],
          [
           "5",
           "0",
           "0",
           "0"
          ]
         ],
         "hot": [
          "4,3"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      }
     ]
    },
    {
     "title": "Row 'e'",
     "description": "No match: 1 + min of three neighbours.",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "1",
         "2",
         "3"
        ],
        [
         "2",
         "2",
         "1",
         "2"
        ],
        [
         "3",
         "2",
         "2",
         "2"
        ],
        [
         "4",
         "3",
         "3",
         "2"
        ],
        [
         "5",
         "4",
         "4",
         "3"
        ]
       ],
       "hot": [],
       "marks": {},
       "kind": "grid"
      }
     ],
     "sub": [
      {
       "description": "'e' ≠ 'r' → 1 + min(up 3, left 5, diag 4) = 4",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "3",
           "3",
           "2"
          ],
          [
           "5",
           "4",
           "0",
           "0"
          ]
         ],
         "hot": [
          "5,1"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'e' ≠ 'o' → 1 + min(up 3, left 4, diag 3) = 4",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "3",
           "3",
           "2"
          ],
          [
           "5",
           "4",
           "4",
           "0"
          ]
         ],
         "hot": [
          "5,2"
         ],
         "marks": {},
         "kind": "grid"
        }
       ]
      },
      {
       "description": "'e' ≠ 's' → 1 + min(up 2, left 4, diag 3) = 3",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "dp",
         "cells": [
          [
           "0",
           "1",
           "2",
           "3"
          ],
          [
           "1",
           "1",
           "2",
           "3"
          ],
          [
           "2",
           "2",
           "1",
           "2"
          ],
          [
           "3",
           "2",
           "2",
           "2"
          ],
          [
           "4",
           "3",
           "3",
           "2"
          ],
          [
           "5",
           "4",
           "4",
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
    },
    {
     "title": "Return",
     "description": "dp[5][3] = 3 edits.",
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
         "1",
         "2",
         "3"
        ],
        [
         "1",
         "1",
         "2",
         "3"
        ],
        [
         "2",
         "2",
         "1",
         "2"
        ],
        [
         "3",
         "2",
         "2",
         "2"
        ],
        [
         "4",
         "3",
         "3",
         "2"
        ],
        [
         "5",
         "4",
         "4",
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
 ],
 "why": "horse → rorse → rose → ros: 3 edits.",
 "idea": "Fill a table of edit distances between prefixes."
};
