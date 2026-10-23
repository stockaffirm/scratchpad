window.PROBLEM = {
 "id": "76",
 "title": "Minimum Window Substring",
 "level": "Hard",
 "topic": "Sliding Window",
 "question": "Find the shortest piece of s that contains every letter of t (with repeats).",
 "why": "s = ADOBECODEBANC, t = ABC → the shortest piece is BANC.",
 "idea": "Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest.",
 "input": "s=\"ADOBECODEBANC\", t=\"ABC\"",
 "output": "\"BANC\"",
 "notes": [],
 "best": "Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minWindow(self, s, t):\n        need, missing = Counter(t), len(t)\n        l = 0; best = (0, float('inf'))\n        for r, ch in enumerate(s):\n            if need[ch] > 0: missing -= 1\n            need[ch] -= 1\n            while missing == 0:\n                if r - l < best[1] - best[0]: best = (l, r)\n                need[s[l]] += 1\n                if need[s[l]] > 0: missing += 1\n                l += 1\n        return '' if best[1] == float('inf') else s[best[0]:best[1]+1]",
   "time": "O(n+m)",
   "space": "O(k)",
   "explain": "Expand right until all needed letters are covered, then shrink left as far as possible; remember the smallest.",
   "frames": [
    {
     "title": "Setup",
     "description": "Need A, B, C.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "need",
       "cells": [
        "A",
        "B",
        "C"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Window found at r = 5",
     "description": "Valid window 'ADOBEC' (length 6) — best so far.",
     "codeLines": [
      9,
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {
        "0": "l",
        "5": "r"
       }
      }
     ],
     "sub": [
      {
       "description": "add 'A' → it was needed, missing = 2",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 2
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0
         ],
         "marks": {
          "0": "l,r"
         }
        }
       ]
      },
      {
       "description": "add 'D' → not needed (or extra), missing = 2",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 2
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0,
          1
         ],
         "marks": {
          "0": "l",
          "1": "r"
         }
        }
       ]
      },
      {
       "description": "add 'O' → not needed (or extra), missing = 2",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 2
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0,
          1,
          2
         ],
         "marks": {
          "0": "l",
          "2": "r"
         }
        }
       ]
      },
      {
       "description": "add 'B' → it was needed, missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0,
          1,
          2,
          3
         ],
         "marks": {
          "0": "l",
          "3": "r"
         }
        }
       ]
      },
      {
       "description": "add 'E' → not needed (or extra), missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0,
          1,
          2,
          3,
          4
         ],
         "marks": {
          "0": "l",
          "4": "r"
         }
        }
       ]
      },
      {
       "description": "add 'C' → it was needed, missing = 0",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0,
          1,
          2,
          3,
          4,
          5
         ],
         "marks": {
          "0": "l",
          "5": "r"
         }
        }
       ]
      },
      {
       "description": "all letters covered! window 'ADOBEC' (length 6) is the best so far",
       "codeLines": [
        9
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          0,
          1,
          2,
          3,
          4,
          5
         ],
         "marks": {
          "0": "l",
          "5": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'A' from the left → it was needed, missing = 1",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5
         ],
         "marks": {
          "1": "l",
          "5": "r"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Window found at r = 12",
     "description": "Valid window 'BANC' (length 4) — best so far.",
     "codeLines": [
      9,
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [
        9,
        10,
        11,
        12
       ],
       "marks": {
        "9": "l",
        "12": "r"
       }
      }
     ],
     "sub": [
      {
       "description": "add 'O' → not needed (or extra), missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5,
          6
         ],
         "marks": {
          "1": "l",
          "6": "r"
         }
        }
       ]
      },
      {
       "description": "add 'D' → not needed (or extra), missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5,
          6,
          7
         ],
         "marks": {
          "1": "l",
          "7": "r"
         }
        }
       ]
      },
      {
       "description": "add 'E' → not needed (or extra), missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8
         ],
         "marks": {
          "1": "l",
          "8": "r"
         }
        }
       ]
      },
      {
       "description": "add 'B' → not needed (or extra), missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9
         ],
         "marks": {
          "1": "l",
          "9": "r"
         }
        }
       ]
      },
      {
       "description": "add 'A' → it was needed, missing = 0",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "1": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "window 'DOBECODEBA' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "1": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'D' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "2": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "window 'OBECODEBA' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "2": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'O' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "3": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "window 'BECODEBA' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "3": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'B' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "4": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "window 'ECODEBA' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          4,
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "4": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'E' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "5": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "window 'CODEBA' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          5,
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "5": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'C' from the left → it was needed, missing = 1",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          6,
          7,
          8,
          9,
          10
         ],
         "marks": {
          "6": "l",
          "10": "r"
         }
        }
       ]
      },
      {
       "description": "add 'N' → not needed (or extra), missing = 1",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          6,
          7,
          8,
          9,
          10,
          11
         ],
         "marks": {
          "6": "l",
          "11": "r"
         }
        }
       ]
      },
      {
       "description": "add 'C' → it was needed, missing = 0",
       "codeLines": [
        6
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          6,
          7,
          8,
          9,
          10,
          11,
          12
         ],
         "marks": {
          "6": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "window 'ODEBANC' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          6,
          7,
          8,
          9,
          10,
          11,
          12
         ],
         "marks": {
          "6": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'O' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          7,
          8,
          9,
          10,
          11,
          12
         ],
         "marks": {
          "7": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "window 'DEBANC' is covered but not shorter than the best",
       "codeLines": [
        8
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          7,
          8,
          9,
          10,
          11,
          12
         ],
         "marks": {
          "7": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'D' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          8,
          9,
          10,
          11,
          12
         ],
         "marks": {
          "8": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "all letters covered! window 'EBANC' (length 5) is the best so far",
       "codeLines": [
        9
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          8,
          9,
          10,
          11,
          12
         ],
         "marks": {
          "8": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'E' from the left",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          9,
          10,
          11,
          12
         ],
         "marks": {
          "9": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "all letters covered! window 'BANC' (length 4) is the best so far",
       "codeLines": [
        9
       ],
       "facts": {
        "missing": 0
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          9,
          10,
          11,
          12
         ],
         "marks": {
          "9": "l",
          "12": "r"
         }
        }
       ]
      },
      {
       "description": "shrink: drop 'B' from the left → it was needed, missing = 1",
       "codeLines": [
        10,
        12
       ],
       "facts": {
        "missing": 1
       },
       "rows": [
        {
         "name": "s",
         "cells": [
          "A",
          "D",
          "O",
          "B",
          "E",
          "C",
          "O",
          "D",
          "E",
          "B",
          "A",
          "N",
          "C"
         ],
         "hot": [
          10,
          11,
          12
         ],
         "marks": {
          "10": "l",
          "12": "r"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "Smallest window: 'BANC'.",
     "codeLines": [
      13
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "A",
        "D",
        "O",
        "B",
        "E",
        "C",
        "O",
        "D",
        "E",
        "B",
        "A",
        "N",
        "C"
       ],
       "hot": [
        9,
        10,
        11,
        12
       ],
       "marks": {
        "9": "l",
        "12": "r"
       }
      }
     ]
    }
   ]
  }
 ]
};
