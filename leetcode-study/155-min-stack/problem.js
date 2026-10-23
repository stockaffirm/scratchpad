window.PROBLEM = {
 "id": "155",
 "title": "Min Stack",
 "level": "Medium",
 "topic": "Stack",
 "question": "A stack that can also tell you its smallest value instantly.",
 "why": "push -2, 0, -3 → getMin = -3 → pop → top = 0 → getMin = -2.",
 "idea": "Store each value together with the minimum so far.",
 "input": "push(-2), push(0), push(-3), getMin(), pop(), top(), getMin()",
 "output": "-3, 0, -2",
 "notes": [],
 "best": "Store each value together with the minimum so far.",
 "variants": [
  {
   "name": "Solution",
   "code": "class MinStack:\n    def __init__(self): self.st = []\n    def push(self, v):\n        self.st.append((v, min(v, self.st[-1][1] if self.st else v)))\n    def pop(self): self.st.pop()\n    def top(self): return self.st[-1][0]\n    def getMin(self): return self.st[-1][1]",
   "time": "O(1) per op",
   "space": "O(n)",
   "explain": "Store each value together with the minimum so far.",
   "frames": [
    {
     "title": "push(-2)",
     "description": "min so far = -2.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "(value, min)",
       "cells": [
        "(-2,-2)"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "push(0)",
     "description": "min stays -2.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "(value, min)",
       "cells": [
        "(-2,-2)",
        "(0,-2)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "push(-3)",
     "description": "new min = -3.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "(value, min)",
       "cells": [
        "(-2,-2)",
        "(0,-2)",
        "(-3,-3)"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "getMin()",
     "description": "Top's min → -3.",
     "codeLines": [
      7
     ],
     "facts": {
      "returns": -3
     },
     "rows": [
      {
       "name": "(value, min)",
       "cells": [
        "(-2,-2)",
        "(0,-2)",
        "(-3,-3)"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "pop()",
     "description": "Remove (-3,-3).",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "(value, min)",
       "cells": [
        "(-2,-2)",
        "(0,-2)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "top() / getMin()",
     "description": "top = 0, min = -2.",
     "codeLines": [
      6
     ],
     "facts": {
      "top": 0,
      "min": -2
     },
     "rows": [
      {
       "name": "(value, min)",
       "cells": [
        "(-2,-2)",
        "(0,-2)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
