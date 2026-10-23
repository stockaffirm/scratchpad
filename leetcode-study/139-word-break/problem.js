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
     ],
     "sub": [
      {
       "description": "i = 1: 'l' ✗ → dp[1] = False",
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
          0
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
          "0",
          "0",
          "0",
          "0",
          "0"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "i = 2: 'le' ✗, [dp[1] false] 'e' ✗ → dp[2] = False",
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
          1
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
          "0",
          "0",
          "0",
          "0",
          "0"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "i = 3: 'lee' ✗, [dp[1] false] 'ee' ✗, [dp[2] false] 'e' ✗ → dp[3] = False",
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
          2
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
          "0",
          "0",
          "0",
          "0",
          "0"
         ],
         "hot": [
          3
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "i = 4: 'leet' ✓, [dp[1] false] 'eet' ✗, [dp[2] false] 'et' ✗, [dp[3] false] 't' ✗ → dp[4] = True",
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
     ],
     "sub": [
      {
       "description": "i = 5: [dp[1] false] 'eetc' ✗, [dp[2] false] 'etc' ✗, [dp[3] false] 'tc' ✗, 'c' ✗ → dp[5] = False",
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
          3,
          4
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
          5
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "i = 6: [dp[2] false] 'etco' ✗, [dp[3] false] 'tco' ✗, 'co' ✗, [dp[5] false] 'o' ✗ → dp[6] = False",
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
          3,
          4,
          5
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
          6
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "i = 7: [dp[3] false] 'tcod' ✗, 'cod' ✗, [dp[5] false] 'od' ✗, [dp[6] false] 'd' ✗ → dp[7] = False",
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
          3,
          4,
          5,
          6
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
          7
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "i = 8: 'code' ✓, [dp[5] false] 'ode' ✗, [dp[6] false] 'de' ✗, [dp[7] false] 'e' ✗ → dp[8] = True",
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
          3,
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
