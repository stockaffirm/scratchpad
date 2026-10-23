window.PROBLEM = {
 "id": "12",
 "title": "Integer to Roman",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Convert a number to a Roman numeral.",
 "why": "1994 = M (1000) + CM (900) + XC (90) + IV (4) → MCMXCIV.",
 "idea": "Go through the values from biggest to smallest, taking each as many times as it fits.",
 "input": "1994",
 "output": "\"MCMXCIV\"",
 "notes": [],
 "best": "Go through the values from biggest to smallest, taking each as many times as it fits.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def intToRoman(self, num):\n        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n        res = ''\n        for v, s in vals:\n            while num >= v:\n                res += s; num -= v\n        return res",
   "time": "O(1)",
   "space": "O(1)",
   "explain": "Go through the values from biggest to smallest, taking each as many times as it fits.",
   "frames": [
    {
     "title": "Start",
     "description": "Remaining number 1994.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "remaining",
       "cells": [
        "1994"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take M (1000)",
     "description": "Remaining 994.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "remaining",
       "cells": [
        "994"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "M"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take CM (900)",
     "description": "Remaining 94.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "remaining",
       "cells": [
        "94"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "MCM"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take XC (90)",
     "description": "Remaining 4.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "remaining",
       "cells": [
        "4"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "MCMXC"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Take IV (4)",
     "description": "Remaining 0.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "remaining",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "MCMXCIV"
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
     "description": "MCMXCIV",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "M",
        "C",
        "M",
        "X",
        "C",
        "I",
        "V"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
