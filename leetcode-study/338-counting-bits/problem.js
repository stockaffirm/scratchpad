window.PROBLEM = {
 "id": "338",
 "title": "Counting Bits",
 "level": "Easy",
 "topic": "Bit Manipulation",
 "question": "For every i in 0…n, count the 1-bits in i's binary form.",
 "input": "n = 5",
 "output": "[0, 1, 1, 2, 1, 2]",
 "notes": [
  "i >> 1 drops the last bit.",
  "i & 1 is that last bit."
 ],
 "best": "DP on bits: ans[i] = ans[i >> 1] + (i & 1).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def countBits(self, n):\n        ans = [0] * (n + 1)\n        for i in range(1, n + 1):\n            ans[i] = ans[i >> 1] + (i & 1)\n        return ans",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "i has the bits of i/2 plus its last bit.",
   "frames": [
    {
     "title": "Start",
     "description": "ans[0] = 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "i",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "ans",
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
     "title": "i = 1 (1)",
     "description": "ans[0] + 1 = 1",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "i",
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
       "name": "ans",
       "cells": [
        "0",
        "1"
       ],
       "hot": [
        1,
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 2 (10)",
     "description": "ans[1] + 0 = 1",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "i",
       "cells": [
        "0",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "0",
        "1",
        "1"
       ],
       "hot": [
        2,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 3 (11)",
     "description": "ans[1] + 1 = 2",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "i",
       "cells": [
        "0",
        "1",
        "2",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "0",
        "1",
        "1",
        "2"
       ],
       "hot": [
        3,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 4 (100)",
     "description": "ans[2] + 0 = 1",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "i",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "1"
       ],
       "hot": [
        4,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 5 (101)",
     "description": "ans[2] + 1 = 2",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "i",
       "cells": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        5,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "[0,1,1,2,1,2]",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "ans",
       "cells": [
        "0",
        "1",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
