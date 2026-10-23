window.PROBLEM = {
 "id": "121",
 "title": "Best Time to Buy and Sell Stock",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Prices per day. Buy once, sell once later. What's the biggest profit (0 if none)?",
 "why": "[7,1,5,3,6,4]: buy at 1, sell at 6 → profit 5.",
 "idea": "Track the lowest price so far; profit today = price − lowest.",
 "input": "[7,1,5,3,6,4]",
 "output": "5",
 "notes": [],
 "best": "Track the lowest price so far; profit today = price − lowest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxProfit(self, prices):\n        low, best = float('inf'), 0\n        for p in prices:\n            low = min(low, p)\n            best = max(best, p - low)\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Track the lowest price so far; profit today = price − lowest.",
   "frames": [
    {
     "title": "Start",
     "description": "Nothing bought yet.",
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
     "title": "Day 0 (price 7)",
     "description": "lowest so far 7; profit if sold now 0",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "best": 0
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
        0
       ],
       "marks": {}
      },
      {
       "name": "low",
       "cells": [
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 1 (price 1)",
     "description": "lowest so far 1; profit if sold now 0",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "best": 0
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
        1
       ],
       "marks": {}
      },
      {
       "name": "low",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 2 (price 5)",
     "description": "lowest so far 1; profit if sold now 4",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "best": 4
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
        2
       ],
       "marks": {}
      },
      {
       "name": "low",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 3 (price 3)",
     "description": "lowest so far 1; profit if sold now 2",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "best": 4
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
        3
       ],
       "marks": {}
      },
      {
       "name": "low",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 4 (price 6)",
     "description": "lowest so far 1; profit if sold now 5",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "best": 5
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
        4
       ],
       "marks": {}
      },
      {
       "name": "low",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 5 (price 4)",
     "description": "lowest so far 1; profit if sold now 3",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "best": 5
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
        5
       ],
       "marks": {}
      },
      {
       "name": "low",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Best profit = 5.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "5"
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
