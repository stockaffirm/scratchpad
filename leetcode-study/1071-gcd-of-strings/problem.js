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
     ],
     "sub": [
      {
       "description": "k = 6: 9 % 6 = 3 ≠ 0 → can't repeat evenly, skip",
       "codeLines": [
        5
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
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "k = 5: 9 % 5 = 4 ≠ 0 → can't repeat evenly, skip",
       "codeLines": [
        5
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
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "k = 4: 9 % 4 = 1 ≠ 0 → can't repeat evenly, skip",
       "codeLines": [
        5
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
          2
         ],
         "marks": {}
        }
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
     ],
     "sub": [
      {
       "description": "3 copies[0] = A",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          0
         ]
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
       "description": "3 copies[1] = B",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          1
         ]
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
       "description": "3 copies[2] = C",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          2
         ]
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
       "description": "3 copies[3] = A",
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
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          3
         ]
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
       "description": "3 copies[4] = B",
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
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          4
         ]
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
       "description": "3 copies[5] = C",
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
          "·",
          "·",
          "·"
         ],
         "hot": [
          5
         ]
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
       "description": "3 copies[6] = A",
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
          "·",
          "·"
         ],
         "hot": [
          6
         ]
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
       "description": "3 copies[7] = B",
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
          "·"
         ],
         "hot": [
          7
         ]
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
       "description": "3 copies[8] = C. Base ABC; repeat it 3 times for str1 and 2 times for str2.",
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
         "hot": [
          8
         ]
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
     ],
     "sub": [
      {
       "description": "str1+str2[0] = A",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          0
         ]
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
       "description": "str1+str2[1] = B",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          1
         ]
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
       "description": "str1+str2[2] = C",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          2
         ]
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
       "description": "str1+str2[3] = A",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          3
         ]
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
       "description": "str1+str2[4] = B",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          4
         ]
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
       "description": "str1+str2[5] = C",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          5
         ]
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
       "description": "str1+str2[6] = A",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          6
         ]
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
       "description": "str1+str2[7] = B",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          7
         ]
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
       "description": "str1+str2[8] = C",
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
          "·",
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          8
         ]
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
       "description": "str1+str2[9] = A",
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
          "·",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          9
         ]
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
       "description": "str1+str2[10] = B",
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
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          10
         ]
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
       "description": "str1+str2[11] = C",
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
          "·",
          "·",
          "·"
         ],
         "hot": [
          11
         ]
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
       "description": "str1+str2[12] = A",
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
          "·",
          "·"
         ],
         "hot": [
          12
         ]
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
       "description": "str1+str2[13] = B",
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
          "·"
         ],
         "hot": [
          13
         ]
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
       "description": "str1+str2[14] = C. Concatenating in either order gives exactly the same string.",
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
         "hot": [
          14
         ]
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
