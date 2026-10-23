window.PROBLEM = {
 "id": "392",
 "title": "Is Subsequence",
 "level": "Easy",
 "topic": "Two Pointers",
 "question": "Is s a subsequence of t (same order, gaps allowed)?",
 "input": "s=\"ace\", t=\"abcde\"",
 "output": "True",
 "notes": [
  "Never move backwards in either string.",
  "Match greedily: the earliest match is always safe."
 ],
 "best": "One scan of t with a pointer into s.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isSubsequence(self, s, t):\n        i = 0\n        for ch in t:\n            if i < len(s) and ch == s[i]:\n                i += 1\n        return i == len(s)",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Advance in s only when t's letter matches.",
   "frames": [
    {
     "title": "Input",
     "description": "Find a, c, e in order inside t.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
       ],
       "hot": [],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "t",
       "cells": [
        "a",
        "b",
        "c",
        "d",
        "e"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "t[0] = a",
     "description": "Match! Move i to 1.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
       ],
       "hot": [
        0
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "t",
       "cells": [
        "a",
        "b",
        "c",
        "d",
        "e"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "ch"
       }
      }
     ]
    },
    {
     "title": "t[1] = b",
     "description": "No match, keep i.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
       ],
       "hot": [],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "t",
       "cells": [
        "a",
        "b",
        "c",
        "d",
        "e"
       ],
       "hot": [],
       "marks": {
        "1": "ch"
       }
      }
     ]
    },
    {
     "title": "t[2] = c",
     "description": "Match! Move i to 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
       ],
       "hot": [
        1
       ],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "t",
       "cells": [
        "a",
        "b",
        "c",
        "d",
        "e"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "ch"
       }
      }
     ]
    },
    {
     "title": "t[3] = d",
     "description": "No match, keep i.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
       ],
       "hot": [],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "t",
       "cells": [
        "a",
        "b",
        "c",
        "d",
        "e"
       ],
       "hot": [],
       "marks": {
        "3": "ch"
       }
      }
     ]
    },
    {
     "title": "t[4] = e",
     "description": "Match! Move i to 3.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
       ],
       "hot": [
        2
       ],
       "marks": {
        "3": "i"
       }
      },
      {
       "name": "t",
       "cells": [
        "a",
        "b",
        "c",
        "d",
        "e"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "ch"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "i reached len(s) = 3.",
     "codeLines": [
      7
     ],
     "facts": {
      "result": "True"
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "c",
        "e"
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
