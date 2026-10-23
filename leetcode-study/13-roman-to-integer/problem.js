window.PROBLEM = {
 "id": "13",
 "title": "Roman to Integer",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Convert a Roman numeral to a number.",
 "why": "MCMXCIV = 1000 + (1000−100) + (100−10) + (5−1) = 1994.",
 "idea": "If a symbol is smaller than the next one, subtract it; otherwise add it.",
 "input": "\"MCMXCIV\"",
 "output": "1994",
 "notes": [],
 "best": "If a symbol is smaller than the next one, subtract it; otherwise add it.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def romanToInt(self, s):\n        v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n        total = 0\n        for i, ch in enumerate(s):\n            if i + 1 < len(s) and v[ch] < v[s[i+1]]: total -= v[ch]\n            else: total += v[ch]\n        return total",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "If a symbol is smaller than the next one, subtract it; otherwise add it.",
   "frames": [
    {
     "title": "Start",
     "description": "Read left to right.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "M",
        "C",
        "M",
        "X",
        "C",
        "I",
        "V"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "M (1000)",
     "description": "Add.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "total": 1000
     },
     "rows": [
      {
       "name": "s",
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
        0
       ],
       "marks": {
        "0": "i"
       }
      }
     ]
    },
    {
     "title": "C (100)",
     "description": "C is smaller than M → subtract.",
     "codeLines": [
      6,
      6
     ],
     "facts": {
      "total": 900
     },
     "rows": [
      {
       "name": "s",
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
        1
       ],
       "marks": {
        "1": "i"
       }
      }
     ]
    },
    {
     "title": "M (1000)",
     "description": "Add.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "total": 1900
     },
     "rows": [
      {
       "name": "s",
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
        2
       ],
       "marks": {
        "2": "i"
       }
      }
     ]
    },
    {
     "title": "X (10)",
     "description": "X is smaller than C → subtract.",
     "codeLines": [
      6,
      6
     ],
     "facts": {
      "total": 1890
     },
     "rows": [
      {
       "name": "s",
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
        3
       ],
       "marks": {
        "3": "i"
       }
      }
     ]
    },
    {
     "title": "C (100)",
     "description": "Add.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "total": 1990
     },
     "rows": [
      {
       "name": "s",
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
        4
       ],
       "marks": {
        "4": "i"
       }
      }
     ]
    },
    {
     "title": "I (1)",
     "description": "I is smaller than V → subtract.",
     "codeLines": [
      6,
      6
     ],
     "facts": {
      "total": 1989
     },
     "rows": [
      {
       "name": "s",
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
        5
       ],
       "marks": {
        "5": "i"
       }
      }
     ]
    },
    {
     "title": "V (5)",
     "description": "Add.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "total": 1994
     },
     "rows": [
      {
       "name": "s",
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
        6
       ],
       "marks": {
        "6": "i"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "1994",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "1994"
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
