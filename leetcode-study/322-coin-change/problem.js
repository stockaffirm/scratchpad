window.PROBLEM = {
 "id": "322",
 "title": "Coin Change",
 "level": "Medium",
 "topic": "1D DP",
 "question": "With unlimited coins of given values, what's the fewest coins that make the amount? (−1 if impossible)",
 "why": "coins [1,2,5], amount 11 → 5 + 5 + 1 = 3 coins.",
 "idea": "dp[a] = 1 + the best of dp[a − coin] over all coins.",
 "input": "coins=[1,2,5], amount=11",
 "output": "3",
 "notes": [],
 "best": "dp[a] = 1 + the best of dp[a − coin] over all coins.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def coinChange(self, coins, amount):\n        dp = [0] + [float('inf')] * amount\n        for a in range(1, amount + 1):\n            for c in coins:\n                if c <= a: dp[a] = min(dp[a], dp[a - c] + 1)\n        return dp[amount] if dp[amount] != float('inf') else -1",
   "time": "O(amount·coins)",
   "space": "O(amount)",
   "explain": "dp[a] = 1 + the best of dp[a − coin] over all coins.",
   "frames": [
    {
     "title": "Start",
     "description": "dp[a] = fewest coins to make a. dp[0] = 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "amount",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "coins needed",
       "cells": [
        "0",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞",
        "∞"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Small amounts",
     "description": "dp[1]=1, dp[2]=1, dp[3]=2, dp[4]=2, dp[5]=1.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "amount",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "coins needed",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "2",
        "1"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "amount 1: coin 1 → dp[0]+1 = 1 → best 1",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 2: coin 1 → dp[1]+1 = 2, coin 2 → dp[0]+1 = 1 → best 1",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 3: coin 1 → dp[2]+1 = 2, coin 2 → dp[1]+1 = 2 → best 2",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          3
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          3
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 4: coin 1 → dp[3]+1 = 3, coin 2 → dp[2]+1 = 2 → best 2",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          4
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          4
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 5: coin 1 → dp[4]+1 = 3, coin 2 → dp[3]+1 = 3, coin 5 → dp[0]+1 = 1 → best 1",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          5
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          5
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Up to 10",
     "description": "dp[6]=2 (5+1), dp[7]=2 (5+2), dp[8]=3, dp[9]=3, dp[10]=2 (5+5).",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "amount",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "coins needed",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "2",
        "1",
        "2",
        "2",
        "3",
        "3",
        "2"
       ],
       "hot": [
        10
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "amount 6: coin 1 → dp[5]+1 = 2, coin 2 → dp[4]+1 = 3, coin 5 → dp[1]+1 = 2 → best 2",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          6
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "2",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          6
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 7: coin 1 → dp[6]+1 = 3, coin 2 → dp[5]+1 = 2, coin 5 → dp[2]+1 = 2 → best 2",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          7
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "2",
          "2",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          7
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 8: coin 1 → dp[7]+1 = 3, coin 2 → dp[6]+1 = 3, coin 5 → dp[3]+1 = 3 → best 3",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          8
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "2",
          "2",
          "3",
          "·",
          "·",
          "·"
         ],
         "hot": [
          8
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 9: coin 1 → dp[8]+1 = 4, coin 2 → dp[7]+1 = 3, coin 5 → dp[4]+1 = 3 → best 3",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          9
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "2",
          "2",
          "3",
          "3",
          "·",
          "·"
         ],
         "hot": [
          9
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "amount 10: coin 1 → dp[9]+1 = 4, coin 2 → dp[8]+1 = 4, coin 5 → dp[5]+1 = 2 → best 2",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          10
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "2",
          "2",
          "3",
          "3",
          "2",
          "·"
         ],
         "hot": [
          10
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "dp[11]",
     "description": "dp[10] + 1 = 3 (5 + 5 + 1).",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "amount",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "coins needed",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "2",
        "1",
        "2",
        "2",
        "3",
        "3",
        "2",
        "3"
       ],
       "hot": [
        11
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "amount 11: coin 1 → dp[10]+1 = 3, coin 2 → dp[9]+1 = 4, coin 5 → dp[6]+1 = 3 → best 3",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "amount",
         "cells": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11"
         ],
         "hot": [
          11
         ],
         "marks": {}
        },
        {
         "name": "coins needed",
         "cells": [
          "0",
          "1",
          "1",
          "2",
          "2",
          "1",
          "2",
          "2",
          "3",
          "3",
          "2",
          "3"
         ],
         "hot": [
          11
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "3",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "3"
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
