window.PROBLEM = {
 "id": "790",
 "title": "Domino and Tromino Tiling",
 "level": "Medium",
 "topic": "DP - 1D",
 "question": "Count ways to tile a 2×n board with dominoes and L-trominoes (mod 10⁹+7).",
 "input": "n = 5",
 "output": "24",
 "notes": [
  "Trominoes must come in interlocking pairs.",
  "The recurrence comes from counting boundary shapes."
 ],
 "best": "Recurrence dp[n] = 2·dp[n-1] + dp[n-3] (found from the tromino patterns).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def numTilings(self, n):\n        dp = [1, 1, 2]\n        for i in range(3, n + 1):\n            dp.append((2 * dp[-1] + dp[-3]) % (10**9 + 7))\n        return dp[n]",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "dp[n] = 2·dp[n-1] + dp[n-3].",
   "frames": [
    {
     "title": "Base",
     "description": "dp[0]=1, dp[1]=1, dp[2]=2.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "1",
        "1",
        "2"
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
     "title": "dp[3]",
     "description": "2×2 + 1 = 5",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "1",
        "1",
        "2",
        "5"
       ],
       "hot": [
        3,
        2,
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "dp[4]",
     "description": "2×5 + 1 = 11",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "1",
        "1",
        "2",
        "5",
        "11"
       ],
       "hot": [
        4,
        3,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "dp[5]",
     "description": "2×11 + 2 = 24",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "1",
        "1",
        "2",
        "5",
        "11",
        "24"
       ],
       "hot": [
        5,
        4,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "24 tilings for n = 5.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "24"
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
