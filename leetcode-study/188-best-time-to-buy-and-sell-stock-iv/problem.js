window.PROBLEM = {
 "id": "188",
 "title": "Best Time to Buy and Sell Stock IV",
 "level": "Hard",
 "topic": "Multidimensional DP",
 "question": "You may complete at most k buy-sell trades. What's the maximum profit?",
 "why": "k = 2, prices [3,2,6,5,0,3]: buy 2 sell 6 (+4), buy 0 sell 3 (+3) → 7.",
 "idea": "Like the two-trade version, but keep buy[t] and sell[t] for every trade count t up to k.",
 "input": "k=2, prices=[3,2,6,5,0,3]",
 "output": "7",
 "notes": [],
 "best": "Like the two-trade version, but keep buy[t] and sell[t] for every trade count t up to k.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxProfit(self, k, prices):\n        buy = [float('-inf')] * (k + 1); sell = [0] * (k + 1)\n        for p in prices:\n            for t in range(1, k + 1):\n                buy[t] = max(buy[t], sell[t-1] - p)\n                sell[t] = max(sell[t], buy[t] + p)\n        return sell[k]",
   "time": "O(n·k)",
   "space": "O(k)",
   "explain": "Like the two-trade version, but keep buy[t] and sell[t] for every trade count t up to k.",
   "frames": [
    {
     "title": "Start",
     "description": "sell[t] = best profit with t trades.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 0 (price 3)",
     "description": "sell[1] = 0, sell[2] = 0",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "0",
        "0"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 1 (price 2)",
     "description": "sell[1] = 0, sell[2] = 0",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "0",
        "0"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 2 (price 6)",
     "description": "sell[1] = 4, sell[2] = 4",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "4",
        "4"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 3 (price 5)",
     "description": "sell[1] = 4, sell[2] = 4",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "4",
        "4"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 4 (price 0)",
     "description": "sell[1] = 4, sell[2] = 4",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "4",
        "4"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 5 (price 3)",
     "description": "sell[1] = 4, sell[2] = 7",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "2",
        "6",
        "5",
        "0",
        "3"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "sell",
       "cells": [
        "0",
        "4",
        "7"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "sell[2] = 7.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "7"
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
