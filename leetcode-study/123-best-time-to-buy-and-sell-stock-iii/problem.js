window.PROBLEM = {
 "id": "123",
 "title": "Best Time to Buy and Sell Stock III",
 "level": "Hard",
 "topic": "Multidimensional DP",
 "question": "You may complete at most TWO buy-sell trades. What's the maximum profit?",
 "why": "[3,3,5,0,0,3,1,4]: buy 0 sell 3 (+3), buy 1 sell 4 (+3) → 6.",
 "idea": "Track four numbers: best after buy 1, after sell 1, after buy 2, after sell 2.",
 "input": "[3,3,5,0,0,3,1,4]",
 "output": "6",
 "notes": [],
 "best": "Track four numbers: best after buy 1, after sell 1, after buy 2, after sell 2.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxProfit(self, prices):\n        b1 = b2 = float('-inf'); s1 = s2 = 0\n        for p in prices:\n            b1 = max(b1, -p); s1 = max(s1, b1 + p)\n            b2 = max(b2, s1 - p); s2 = max(s2, b2 + p)\n        return s2",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Track four numbers: best after buy 1, after sell 1, after buy 2, after sell 2.",
   "frames": [
    {
     "title": "Start",
     "description": "Four running values.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 0 (price 3)",
     "description": "After trade 1: 0 · after trade 2: 0",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 0,
      "profit2": 0
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 1 (price 3)",
     "description": "After trade 1: 0 · after trade 2: 0",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 0,
      "profit2": 0
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 2 (price 5)",
     "description": "After trade 1: 2 · after trade 2: 2",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 2,
      "profit2": 2
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 3 (price 0)",
     "description": "After trade 1: 2 · after trade 2: 2",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 2,
      "profit2": 2
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 4 (price 0)",
     "description": "After trade 1: 2 · after trade 2: 2",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 2,
      "profit2": 2
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 5 (price 3)",
     "description": "After trade 1: 3 · after trade 2: 5",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 3,
      "profit2": 5
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 6 (price 1)",
     "description": "After trade 1: 3 · after trade 2: 5",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 3,
      "profit2": 5
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        6
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 7 (price 4)",
     "description": "After trade 1: 4 · after trade 2: 6",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "profit1": 4,
      "profit2": 6
     },
     "rows": [
      {
       "name": "prices",
       "cells": [
        "3",
        "3",
        "5",
        "0",
        "0",
        "3",
        "1",
        "4"
       ],
       "hot": [
        7
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Best with two trades = 6.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "6"
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
