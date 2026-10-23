window.PROBLEM = {
 "id": "172",
 "title": "Factorial Trailing Zeroes",
 "level": "Medium",
 "topic": "Math",
 "question": "How many zeros are at the end of n! (n factorial)?",
 "why": "25! ends in 6 zeros.",
 "idea": "Each zero comes from a factor 10 = 2 × 5, and 5s are rarer than 2s. Count the 5s: n/5 + n/25 + …",
 "input": "25",
 "output": "6",
 "notes": [],
 "best": "Each zero comes from a factor 10 = 2 × 5, and 5s are rarer than 2s. Count the 5s: n/5 + n/25 + …",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def trailingZeroes(self, n):\n        count = 0\n        while n:\n            n //= 5; count += n\n        return count",
   "time": "O(log n)",
   "space": "O(1)",
   "explain": "Each zero comes from a factor 10 = 2 × 5, and 5s are rarer than 2s. Count the 5s: n/5 + n/25 + …",
   "frames": [
    {
     "title": "Start",
     "description": "Count factors of 5 in 1…25.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "25"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Multiples of 5",
     "description": "25 ÷ 5 = 5 numbers (5,10,15,20,25) → count 5.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "5"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Multiples of 25",
     "description": "5 ÷ 5 = 1 more (25 has a second 5) → count 6.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "n",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "6",
     "codeLines": [
      6
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
