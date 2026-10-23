window.PROBLEM = {
 "id": "1318",
 "title": "Minimum Flips to Make a OR b Equal to c",
 "level": "Medium",
 "topic": "Bit Manipulation",
 "question": "Fewest bit flips in a and/or b so that (a OR b) == c.",
 "input": "a=2 (010), b=6 (110), c=5 (101)",
 "output": "3",
 "notes": [
  "c bit 0: flip every 1 in a and b.",
  "c bit 1: flip one only if both are 0."
 ],
 "best": "Check each bit: c-bit 0 needs both zero; c-bit 1 needs at least one.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minFlips(self, a, b, c):\n        flips = 0\n        for i in range(32):\n            x, y, z = (a >> i) & 1, (b >> i) & 1, (c >> i) & 1\n            if z == 0: flips += x + y\n            else: flips += 1 if x + y == 0 else 0\n        return flips",
   "time": "O(1)",
   "space": "O(1)",
   "explain": "Decide bit by bit.",
   "frames": [
    {
     "title": "Bits",
     "description": "Compare bit by bit, right to left.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "0",
        "1",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "1",
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "c",
       "cells": [
        "1",
        "0",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Bit 0",
     "description": "a=0, b=0, c=1 → need a 1 → 1 flip.",
     "codeLines": [
      7
     ],
     "facts": {
      "flips": 1
     },
     "rows": [
      {
       "name": "a",
       "cells": [
        "0",
        "1",
        "0"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "1",
        "0"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "c",
       "cells": [
        "1",
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
     "title": "Bit 1",
     "description": "a=1, b=1, c=0 → both must be 0 → 2 flips.",
     "codeLines": [
      6
     ],
     "facts": {
      "flips": 3
     },
     "rows": [
      {
       "name": "a",
       "cells": [
        "0",
        "1",
        "0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "1",
        "0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "c",
       "cells": [
        "1",
        "0",
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
     "title": "Bit 2",
     "description": "a=0, b=1, c=1 → already OK.",
     "codeLines": [
      7
     ],
     "facts": {
      "flips": 3
     },
     "rows": [
      {
       "name": "a",
       "cells": [
        "0",
        "1",
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "1",
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "c",
       "cells": [
        "1",
        "0",
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
     "title": "Return",
     "description": "3 flips.",
     "codeLines": [
      8
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
