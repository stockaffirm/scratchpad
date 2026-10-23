window.PROBLEM = {
 "id": "190",
 "title": "Reverse Bits",
 "level": "Easy",
 "topic": "Bit Manipulation",
 "question": "Reverse the 32 bits of a number.",
 "why": "(Shown with 8 bits.) 00001011 reversed is 11010000.",
 "idea": "Repeat: take the last bit of n and push it onto the end of the result.",
 "input": "00001011 (shown as 8 bits)",
 "output": "11010000",
 "notes": [],
 "best": "Repeat: take the last bit of n and push it onto the end of the result.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def reverseBits(self, n):\n        res = 0\n        for _ in range(32):\n            res = (res << 1) | (n & 1)\n            n >>= 1\n        return res",
   "time": "O(1)",
   "space": "O(1)",
   "explain": "Repeat: take the last bit of n and push it onto the end of the result.",
   "frames": [
    {
     "title": "Start",
     "description": "n = 00001011, result = 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "1",
        "0",
        "1",
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
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
     "title": "Step 1",
     "description": "Last bit of n is 1 → result gets 1.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "0",
        "1"
       ],
       "hot": [
        7
       ],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": [
        7
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Step 2",
     "description": "Last bit 1 → result 11.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "0"
       ],
       "hot": [
        7
       ],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1"
       ],
       "hot": [
        6,
        7
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Step 3",
     "description": "Last bit 0 → result 110.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": [
        7
       ],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "0"
       ],
       "hot": [
        5,
        6,
        7
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Finish",
     "description": "After all bits: 11010000.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "1",
        "1",
        "0",
        "1",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
