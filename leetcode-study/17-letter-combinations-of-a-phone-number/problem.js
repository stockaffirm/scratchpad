window.PROBLEM = {
 "id": "17",
 "title": "Letter Combinations of a Phone Number",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "On a phone keypad 2 = abc, 3 = def, and so on. Given some digits, list every letter combination they could spell.",
 "input": "\"23\"",
 "output": "[\"ad\",\"ae\",\"af\",\"bd\",\"be\",\"bf\",\"cd\",\"ce\",\"cf\"]",
 "notes": [
  "Depth = number of digits.",
  "path + ch builds a new string, so no explicit undo is needed."
 ],
 "best": "Backtracking: one level per digit, one branch per letter.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def letterCombinations(self, digits):\n        if not digits: return []\n        phone = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}\n        res = []\n        def go(i, path):\n            if i == len(digits): res.append(path); return\n            for ch in phone[digits[i]]: go(i + 1, path + ch)\n        go(0, '')\n        return res",
   "time": "O(4ⁿ·n)",
   "space": "O(n)",
   "explain": "Choose one letter per digit, recurse, backtrack.",
   "frames": [
    {
     "title": "Digits",
     "description": "2 → abc, 3 → def",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "digits",
       "cells": [
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pick 'a'",
     "description": "Then try d, e, f for digit 3.",
     "codeLines": [
      8,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "a"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "digit 2 → pick 'a'",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "a"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "·"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'd' → path 'ad' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "a",
          "d"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'e' → path 'ae' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "a",
          "e"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'f' → path 'af' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "a",
          "f"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "all of d, e, f tried → undo 'a'",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Pick 'b'",
     "description": "Then try d, e, f for digit 3.",
     "codeLines": [
      8,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "b"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf"
       ],
       "hot": [
        3,
        4,
        5
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "digit 2 → pick 'b'",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "b"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'd' → path 'bd' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "b",
          "d"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd"
         ],
         "hot": [
          3
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'e' → path 'be' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "b",
          "e"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be"
         ],
         "hot": [
          4
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'f' → path 'bf' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "b",
          "f"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf"
         ],
         "hot": [
          5
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "all of d, e, f tried → undo 'b'",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Pick 'c'",
     "description": "Then try d, e, f for digit 3.",
     "codeLines": [
      8,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "c"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf",
        "cd",
        "ce",
        "cf"
       ],
       "hot": [
        6,
        7,
        8
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "digit 2 → pick 'c'",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "c"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'd' → path 'cd' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "c",
          "d"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf",
          "cd"
         ],
         "hot": [
          6
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'e' → path 'ce' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "c",
          "e"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf",
          "cd",
          "ce"
         ],
         "hot": [
          7
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "digit 3 → pick 'f' → path 'cf' is complete → save",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "path",
         "cells": [
          "c",
          "f"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf",
          "cd",
          "ce",
          "cf"
         ],
         "hot": [
          8
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "all of d, e, f tried → undo 'c'",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "res",
         "cells": [
          "ad",
          "ae",
          "af",
          "bd",
          "be",
          "bf",
          "cd",
          "ce",
          "cf"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "9 combinations.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "ad",
        "ae",
        "af",
        "bd",
        "be",
        "bf",
        "cd",
        "ce",
        "cf"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ],
 "why": "2 → a, b, c and 3 → d, e, f gives 3 × 3 = 9 words.",
 "idea": "Pick one letter per digit, going deeper one digit at a time (recursion)."
};
