window.PROBLEM = {
 "id": "714",
 "title": "Best Time to Buy and Sell Stock with Transaction Fee",
 "level": "Medium",
 "topic": "DP - Multidimensional",
 "question": "Unlimited trades, but pay a fee per sale. Maximum profit?",
 "input": "prices=[1,3,2,8,4,9], fee=2",
 "output": "8",
 "notes": [
  "cash = best profit with no share.",
  "hold = best profit while owning a share."
 ],
 "best": "State machine DP: hold vs cash, update both each day.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxProfit(self, prices, fee):\n        hold, cash = -prices[0], 0\n        for p in prices[1:]:\n            hold, cash = max(hold, cash - p), max(cash, hold + p - fee)\n        return cash",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Two states: holding a share, or holding cash.",
   "frames": [
    {
     "title": "Day 0",
     "description": "Buy at 1: hold = -1, cash = 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "1",
        "3",
        "2",
        "8",
        "4",
        "9"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "hold",
       "cells": [
        "-1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cash",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 1 (price 3)",
     "description": "hold = -1, cash = 0",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "1",
        "3",
        "2",
        "8",
        "4",
        "9"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "hold",
       "cells": [
        "-1",
        "-1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "cash",
       "cells": [
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
     "title": "Day 2 (price 2)",
     "description": "hold = -1, cash = 0",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "1",
        "3",
        "2",
        "8",
        "4",
        "9"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "hold",
       "cells": [
        "-1",
        "-1",
        "-1"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "cash",
       "cells": [
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
     "title": "Day 3 (price 8)",
     "description": "hold = -1, cash = 5",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "1",
        "3",
        "2",
        "8",
        "4",
        "9"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "hold",
       "cells": [
        "-1",
        "-1",
        "-1",
        "-1"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "cash",
       "cells": [
        "0",
        "0",
        "0",
        "5"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 4 (price 4)",
     "description": "hold = 1, cash = 5",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "1",
        "3",
        "2",
        "8",
        "4",
        "9"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "hold",
       "cells": [
        "-1",
        "-1",
        "-1",
        "-1",
        "1"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "cash",
       "cells": [
        "0",
        "0",
        "0",
        "5",
        "5"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 5 (price 9)",
     "description": "hold = 1, cash = 8",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "1",
        "3",
        "2",
        "8",
        "4",
        "9"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "hold",
       "cells": [
        "-1",
        "-1",
        "-1",
        "-1",
        "1",
        "1"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "cash",
       "cells": [
        "0",
        "0",
        "0",
        "5",
        "5",
        "8"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Best profit = 8.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "cash",
       "cells": [
        "0",
        "0",
        "0",
        "5",
        "5",
        "8"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
