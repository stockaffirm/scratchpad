window.PROBLEM = {
 "id": "122",
 "title": "Best Time to Buy and Sell Stock II",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Prices per day. Buy and sell as many times as you like (hold one share at most). Maximum profit?",
 "why": "[7,1,5,3,6,4]: buy 1 sell 5 (+4), buy 3 sell 6 (+3) → 7.",
 "idea": "Grab every upward step: add up each day's gain when the price rose.",
 "input": "[7,1,5,3,6,4]",
 "output": "7",
 "notes": [],
 "best": "Grab every upward step: add up each day's gain when the price rose.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxProfit(self, prices):\n        profit = 0\n        for i in range(1, len(prices)):\n            if prices[i] > prices[i - 1]:\n                profit += prices[i] - prices[i - 1]\n        return profit",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Grab every upward step: add up each day's gain when the price rose.",
   "frames": [
    {
     "title": "Start",
     "description": "Collect every price rise.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "7",
        "1",
        "5",
        "3",
        "6",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 1",
     "description": "Down — no trade.",
     "codeLines": [
      5
     ],
     "facts": {
      "profit": 0
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "7",
        "1",
        "5",
        "3",
        "6",
        "4"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 2",
     "description": "Up 1 → 5: +4",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit": 4
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "7",
        "1",
        "5",
        "3",
        "6",
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
     "title": "Day 3",
     "description": "Down — no trade.",
     "codeLines": [
      5
     ],
     "facts": {
      "profit": 4
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "7",
        "1",
        "5",
        "3",
        "6",
        "4"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 4",
     "description": "Up 3 → 6: +3",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit": 7
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "7",
        "1",
        "5",
        "3",
        "6",
        "4"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 5",
     "description": "Down — no trade.",
     "codeLines": [
      5
     ],
     "facts": {
      "profit": 7
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "7",
        "1",
        "5",
        "3",
        "6",
        "4"
       ],
       "hot": [
        4,
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Total profit = 7.",
     "codeLines": [
      7
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
