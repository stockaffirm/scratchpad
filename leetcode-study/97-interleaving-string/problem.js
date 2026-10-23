window.PROBLEM = {
 "id": "97",
 "title": "Interleaving String",
 "level": "Medium",
 "topic": "Multidimensional DP",
 "question": "Can s3 be made by weaving letters of s1 and s2 together, keeping each string's own order?",
 "why": "'ab' and 'cd' can make 'acbd' (a from s1, c from s2, b from s1, d from s2) → True.",
 "idea": "dp[i][j] = can the first i letters of s1 and first j of s2 make the first i+j letters of s3?",
 "input": "s1=\"ab\", s2=\"cd\", s3=\"acbd\"",
 "output": "True",
 "notes": [],
 "best": "dp[i][j] = can the first i letters of s1 and first j of s2 make the first i+j letters of s3?",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isInterleave(self, s1, s2, s3):\n        if len(s1) + len(s2) != len(s3): return False\n        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]; dp[0][0] = True\n        for i in range(len(s1) + 1):\n            for j in range(len(s2) + 1):\n                if i and dp[i-1][j] and s1[i-1] == s3[i+j-1]: dp[i][j] = True\n                if j and dp[i][j-1] and s2[j-1] == s3[i+j-1]: dp[i][j] = True\n        return dp[-1][-1]",
   "time": "O(n·m)",
   "space": "O(n·m)",
   "explain": "dp[i][j] = can the first i letters of s1 and first j of s2 make the first i+j letters of s3?",
   "frames": [
    {
     "title": "Table",
     "description": "Rows = letters of s1 used, columns = letters of s2 used. 1 = possible.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0"
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
     "title": "Use s1's 'a'",
     "description": "Take 'a' from s1 → matches s3[0] = 'a'.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "0",
         "0"
        ],
        [
         "0",
         "0",
         "0"
        ]
       ],
       "hot": [
        "1,0"
       ],
       "marks": {},
       "kind": "grid"
      }
     ]
    },
    {
     "title": "Then s2's 'c'",
     "description": "Take 'c' from s2 → matches s3[1] = 'c'.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
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
     "title": "Then s1's 'b'",
     "description": "Take 'b' from s1 → matches s3[2] = 'b'.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
         "0",
         "1",
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
     "title": "Then s2's 'd'",
     "description": "Take 'd' from s2 → matches s3[3] = 'd'. Bottom-right is 1 → True.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        [
         "1",
         "0",
         "0"
        ],
        [
         "1",
         "1",
         "0"
        ],
        [
         "0",
         "1",
         "1"
        ]
       ],
       "hot": [
        "2,2"
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
