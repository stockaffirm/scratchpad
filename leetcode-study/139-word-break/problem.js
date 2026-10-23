window.PROBLEM = {
 "id": "139",
 "title": "Word Break",
 "level": "Medium",
 "topic": "1D DP",
 "question": "Can the text be split into words from the dictionary?",
 "why": "'leetcode' = 'leet' + 'code' → True.",
 "idea": "dp[i] = True if some earlier dp[j] is True and s[j:i] is a word.",
 "input": "s=\"leetcode\", wordDict=[\"leet\",\"code\"]",
 "output": "True",
 "notes": [],
 "best": "dp[i] = True if some earlier dp[j] is True and s[j:i] is a word.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def wordBreak(self, s, wordDict):\n        words = set(wordDict); dp = [True] + [False] * len(s)\n        for i in range(1, len(s) + 1):\n            dp[i] = any(dp[j] and s[j:i] in words for j in range(i))\n        return dp[-1]",
   "time": "O(n²)",
   "space": "O(n)",
   "explain": "dp[i] = True if some earlier dp[j] is True and s[j:i] is a word.",
   "frames": [
    {
     "title": "Start",
     "description": "dp[0] = True (empty prefix is fine). dp[i] means s[:i] can be split.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "c",
        "o",
        "d",
        "e"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "dp",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
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
     "title": "i = 4",
     "description": "s[0:4] = 'leet' is a word and dp[0] is True → dp[4] = True.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "c",
        "o",
        "d",
        "e"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      },
      {
       "name": "dp",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "1",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 8",
     "description": "s[4:8] = 'code' is a word and dp[4] is True → dp[8] = True.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "c",
        "o",
        "d",
        "e"
       ],
       "hot": [
        4,
        5,
        6,
        7
       ],
       "marks": {}
      },
      {
       "name": "dp",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "1",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": [
        8
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "dp[8] is True.",
     "codeLines": [
      6
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
