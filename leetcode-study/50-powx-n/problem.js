window.PROBLEM = {
 "id": "50",
 "title": "Pow(x, n)",
 "level": "Medium",
 "topic": "Math",
 "question": "Compute x raised to the power n quickly.",
 "why": "2^10 = 1024.",
 "idea": "Repeated squaring: look at the bits of n; multiply the result by x when the bit is 1, and square x each step.",
 "input": "x=2, n=10",
 "output": "1024",
 "notes": [],
 "best": "Repeated squaring: look at the bits of n; multiply the result by x when the bit is 1, and square x each step.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def myPow(self, x, n):\n        if n < 0: x, n = 1 / x, -n\n        res = 1\n        while n:\n            if n & 1: res *= x\n            x *= x; n >>= 1\n        return res",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Repeated squaring: look at the bits of n; multiply the result by x when the bit is 1, and square x each step.",
   "frames": [
    {
     "title": "Start",
     "description": "n = 10 = 1010 in binary.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "n bits",
       "cells": [
        "1",
        "0",
        "1",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "x",
       "cells": [
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Bit 0 = 0",
     "description": "Skip; x = 4.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "n bits",
       "cells": [
        "1",
        "0",
        "1"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "x",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Bit = 1",
     "description": "res = 1 × 4 = 4; x = 16.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "n bits",
       "cells": [
        "1",
        "0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "x",
       "cells": [
        "16"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Bit = 0",
     "description": "Skip; x = 256.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "n bits",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "x",
       "cells": [
        "256"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Bit = 1",
     "description": "res = 4 × 256 = 1024.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "n bits",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "1024"
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
