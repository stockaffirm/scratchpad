window.PROBLEM = {
 "id": "67",
 "title": "Add Binary",
 "level": "Easy",
 "topic": "Bit Manipulation",
 "question": "Add two binary numbers given as strings.",
 "why": "'11' + '1' = 3 + 1 = 4 = '100'.",
 "idea": "Add from the right with a carry, like adding on paper.",
 "input": "a=\"11\", b=\"1\"",
 "output": "\"100\"",
 "notes": [],
 "best": "Add from the right with a carry, like adding on paper.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def addBinary(self, a, b):\n        i, j, carry, res = len(a) - 1, len(b) - 1, 0, []\n        while i >= 0 or j >= 0 or carry:\n            s = carry + (int(a[i]) if i >= 0 else 0) + (int(b[j]) if j >= 0 else 0)\n            res.append(str(s % 2)); carry = s // 2; i -= 1; j -= 1\n        return ''.join(reversed(res))",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Add from the right with a carry, like adding on paper.",
   "frames": [
    {
     "title": "Start",
     "description": "Add digit by digit from the right.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "0",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "1 + 1",
     "description": "= 2 → write 0, carry 1.",
     "codeLines": [
      3,
      6
     ],
     "facts": {
      "carry": 1
     },
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "0",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "result (reversed)",
       "cells": [
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "1 + 0 + carry 1",
     "description": "= 2 → write 0, carry 1.",
     "codeLines": [
      3
     ],
     "facts": {
      "carry": 1
     },
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "0",
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "result (reversed)",
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
     "title": "Carry only",
     "description": "1 → write 1.",
     "codeLines": [
      4
     ],
     "facts": {
      "carry": 0
     },
     "rows": [
      {
       "name": "result (reversed)",
       "cells": [
        "0",
        "0",
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
     "title": "Return",
     "description": "Reverse → '100'.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1",
        "0",
        "0"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
