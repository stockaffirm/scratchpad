window.PROBLEM = {
 "id": "901",
 "title": "Online Stock Span",
 "level": "Medium",
 "topic": "Monotonic Stack",
 "question": "For each price, count consecutive previous days (including today) with price ≤ today's.",
 "input": "100, 80, 60, 70, 60, 75, 85",
 "output": "1, 1, 1, 2, 1, 4, 6",
 "notes": [
  "Popped entries are absorbed into the span.",
  "Stack stays strictly decreasing in price."
 ],
 "best": "Monotonic stack of (price, span) pairs.",
 "variants": [
  {
   "name": "Solution",
   "code": "class StockSpanner:\n    def __init__(self):\n        self.stack = []\n    def next(self, price):\n        span = 1\n        while self.stack and self.stack[-1][0] <= price:\n            span += self.stack.pop()[1]\n        self.stack.append((price, span))\n        return span",
   "time": "O(1) amortised",
   "space": "O(n)",
   "explain": "Merge smaller days into one (price, span) entry.",
   "frames": [
    {
     "title": "Start",
     "description": "Stack of (price, span) pairs.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "price 100",
     "description": "span = 1",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "price 80",
     "description": "span = 1",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)",
        "(80,1)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "price 60",
     "description": "span = 1",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)",
        "(80,1)",
        "(60,1)"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1",
        "1",
        "1"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "price 70",
     "description": "span = 2",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)",
        "(80,1)",
        "(70,2)"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1",
        "1",
        "1",
        "2"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "price 60",
     "description": "span = 1",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)",
        "(80,1)",
        "(70,2)",
        "(60,1)"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1",
        "1",
        "1",
        "2",
        "1"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "price 75",
     "description": "span = 4",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)",
        "(80,1)",
        "(75,4)"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1",
        "1",
        "1",
        "2",
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
     "title": "price 85",
     "description": "span = 6",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "prices",
       "cells": [
        "100",
        "80",
        "60",
        "70",
        "60",
        "75",
        "85"
       ],
       "hot": [
        6
       ],
       "marks": {}
      },
      {
       "name": "stack (price,span)",
       "cells": [
        "(100,1)",
        "(85,6)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "spans",
       "cells": [
        "1",
        "1",
        "1",
        "2",
        "1",
        "4",
        "6"
       ],
       "hot": [
        6
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
