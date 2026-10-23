window.PROBLEM = {
 "id": "191",
 "title": "Number of 1 Bits",
 "level": "Easy",
 "topic": "Bit Manipulation",
 "question": "Count the 1s in the binary form of a number.",
 "why": "11 is 1011 → three 1s.",
 "idea": "n & (n−1) removes the lowest 1 bit. Count how many times you can do it.",
 "input": "11 (1011)",
 "output": "3",
 "notes": [],
 "best": "n & (n−1) removes the lowest 1 bit. Count how many times you can do it.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def hammingWeight(self, n):\n        count = 0\n        while n:\n            n &= n - 1; count += 1\n        return count",
   "time": "O(#ones)",
   "space": "O(1)",
   "explain": "n & (n−1) removes the lowest 1 bit. Count how many times you can do it.",
   "frames": [
    {
     "title": "Start",
     "description": "n = 1011",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "0",
        "1",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Clear a bit",
     "description": "1011 & 1010 = 1010 → count 1.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "0",
        "1",
        "0"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Clear a bit",
     "description": "1010 & 1001 = 1000 → count 2.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1",
        "0",
        "0",
        "0"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Clear a bit",
     "description": "1000 & 0111 = 0000 → count 3.",
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
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "3",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "3"
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
