window.PROBLEM = {
 "id": "150",
 "title": "Evaluate Reverse Polish Notation",
 "level": "Medium",
 "topic": "Stack",
 "question": "Evaluate math written with the operator AFTER its numbers (e.g. 2 1 + means 2 + 1).",
 "why": "['2','1','+','3','*'] = (2 + 1) × 3 = 9.",
 "idea": "Push numbers; at an operator pop two, compute, push the result.",
 "input": "[\"2\",\"1\",\"+\",\"3\",\"*\"]",
 "output": "9",
 "notes": [],
 "best": "Push numbers; at an operator pop two, compute, push the result.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def evalRPN(self, tokens):\n        st = []\n        for t in tokens:\n            if t in '+-*/':\n                b, a = st.pop(), st.pop()\n                st.append(int(a + b if t == '+' else a - b if t == '-' else a * b if t == '*' else a / b))\n            else: st.append(int(t))\n        return st[0]",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Push numbers; at an operator pop two, compute, push the result.",
   "frames": [
    {
     "title": "Start",
     "description": "Empty stack.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "tokens",
       "cells": [
        "2",
        "1",
        "+",
        "3",
        "*"
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
     "title": "'2'",
     "description": "Number → push.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "tokens",
       "cells": [
        "2",
        "1",
        "+",
        "3",
        "*"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'1'",
     "description": "Number → push.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "tokens",
       "cells": [
        "2",
        "1",
        "+",
        "3",
        "*"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "2",
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
     "title": "'+'",
     "description": "2 + 1 = 3",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "tokens",
       "cells": [
        "2",
        "1",
        "+",
        "3",
        "*"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'3'",
     "description": "Number → push.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "tokens",
       "cells": [
        "2",
        "1",
        "+",
        "3",
        "*"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "3",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'*'",
     "description": "3 * 3 = 9",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "tokens",
       "cells": [
        "2",
        "1",
        "+",
        "3",
        "*"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "9"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "9",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "9"
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
