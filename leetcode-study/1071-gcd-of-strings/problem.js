window.PROBLEM = {
 "id": "1071",
 "title": "Greatest Common Divisor of Strings",
 "level": "Easy",
 "topic": "Strings",
 "question": "A string 'divides' another if repeating it enough times builds the other one. Find the longest string that divides BOTH words.",
 "input": "\"ABCABCABC\", \"ABCABC\"",
 "output": "\"ABC\"",
 "notes": [
  "n1 = len(str1)//k and n2 = len(str2)//k are the repetition counts.",
  "String multiplication repeats characters: 'ABC' * 3 becomes 'ABCABCABC'.",
  "Strings concatenation + GCD avoids checking candidate lengths one by one.",
  "String slicing and repetition cost time and allocate memory proportional to their length."
 ],
 "best": "Concatenate + GCD is better: O(n+m) time instead of an expensive candidate scan.",
 "variants": [
  {
   "name": "Candidate scan",
   "code": "class Solution:\n    def gcdOfStrings(self, str1, str2):\n        len1, len2 = len(str1), len(str2)\n        def valid(k):\n            if len1 % k or len2 % k: return False\n            n1, n2 = len1 // k, len2 // k\n            base = str1[:k]\n            return str1 == n1*base and str2 == n2*base\n        for i in range(min(len1, len2), 0, -1):\n            if valid(i): return str1[:i]\n        return ''",
   "time": "O(min(n,m) × (n+m)) upper bound",
   "space": "O(n+m)",
   "explain": "Straightforward verification but revisits many candidates.",
   "frames": [
    {
     "title": "Input",
     "description": "Find the longest string that can repeat to form both inputs.",
     "codeLines": [
      2,
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "str1",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      },
      {
       "name": "str2",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Check k = 6, 5, 4",
     "description": "At least one length is not divisible by each candidate, so skip.",
     "codeLines": [
      5,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "candidates",
       "cells": [
        "6",
        "5",
        "4"
       ],
       "hot": [
        0,
        1,
        2
       ]
      }
     ]
    },
    {
     "title": "Check k = 3",
     "description": "Base ABC; repeat it 3 times for str1 and 2 times for str2.",
     "codeLines": [
      6,
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "base",
       "cells": [
        "A",
        "B",
        "C"
       ],
       "hot": [
        0,
        1,
        2
       ]
      },
      {
       "name": "3 copies",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      },
      {
       "name": "2 copies",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Return",
     "description": "Both reconstructions match, so the answer is ABC.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "A",
        "B",
        "C"
       ],
       "hot": [
        0,
        1,
        2
       ]
      }
     ]
    }
   ]
  },
  {
   "name": "Concatenate + GCD",
   "code": "from math import gcd\nclass Solution:\n    def gcdOfStrings(self, str1, str2):\n        if str1 + str2 != str2 + str1:\n            return ''\n        max_length = gcd(len(str1), len(str2))\n        return str1[:max_length]",
   "time": "O(n+m)",
   "space": "O(n+m)",
   "explain": "Compact mathematical solution with a single linear compatibility check.",
   "frames": [
    {
     "title": "Input",
     "description": "Both strings are repetitions of ABC.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "str1",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      },
      {
       "name": "str2",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Commutativity check",
     "description": "Concatenating in either order gives exactly the same string.",
     "codeLines": [
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "str1+str2",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      },
      {
       "name": "str2+str1",
       "cells": [
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C",
        "A",
        "B",
        "C"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Calculate length",
     "description": "gcd(9,6) = 3, so the longest valid base has length 3.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "lengths",
       "cells": [
        "9",
        "6"
       ],
       "hot": []
      },
      {
       "name": "GCD",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ]
      }
     ]
    },
    {
     "title": "Return prefix",
     "description": "Take str1[:3].",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "A",
        "B",
        "C"
       ],
       "hot": [
        0,
        1,
        2
       ]
      }
     ]
    }
   ]
  }
 ],
 "why": "'ABCABCABC' is ABC three times. 'ABCABC' is ABC two times. So ABC divides both.",
 "idea": "If str1+str2 ≠ str2+str1 there is no answer. Otherwise the answer's length is gcd(len1, len2)."
};
