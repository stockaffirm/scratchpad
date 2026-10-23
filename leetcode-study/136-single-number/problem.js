window.PROBLEM = {
 "id": "136",
 "title": "Single Number",
 "level": "Easy",
 "topic": "Bit Manipulation",
 "question": "Every number appears twice except one. Find it in O(1) space.",
 "input": "[4, 1, 2, 1, 2]",
 "output": "4",
 "notes": [
  "XOR is commutative and associative.",
  "Pairs vanish; the single remains."
 ],
 "best": "XOR everything: x ^ x = 0 and 0 ^ y = y.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def singleNumber(self, nums):\n        res = 0\n        for x in nums: res ^= x\n        return res",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "XOR cancels equal pairs.",
   "frames": [
    {
     "title": "Start",
     "description": "res = 0",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "^ 4",
     "description": "0 XOR 4 = 4",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
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
     "title": "^ 1",
     "description": "4 XOR 1 = 5",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "^ 2",
     "description": "5 XOR 2 = 7",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "^ 1",
     "description": "7 XOR 1 = 6",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "^ 2",
     "description": "6 XOR 2 = 4",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "4",
        "1",
        "2",
        "1",
        "2"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
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
     "title": "Return",
     "description": "Pairs cancelled → 4.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "4"
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
