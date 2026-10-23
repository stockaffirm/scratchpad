window.PROBLEM = {
 "id": "22",
 "title": "Generate Parentheses",
 "level": "Medium",
 "topic": "Backtracking",
 "question": "List every valid way to write n pairs of parentheses.",
 "why": "n = 2 → (()) and ()().",
 "idea": "Build the string: you may add '(' while you have some left, and ')' only if there is an unmatched '('.",
 "input": "n = 2",
 "output": "[\"(())\",\"()()\"]",
 "notes": [],
 "best": "Build the string: you may add '(' while you have some left, and ')' only if there is an unmatched '('.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def generateParenthesis(self, n):\n        res = []\n        def go(s, open_, close):\n            if len(s) == 2 * n: res.append(s); return\n            if open_ < n: go(s + '(', open_ + 1, close)\n            if close < open_: go(s + ')', open_, close + 1)\n        go('', 0, 0)\n        return res",
   "time": "O(4ⁿ/√n)",
   "space": "O(n)",
   "explain": "Build the string: you may add '(' while you have some left, and ')' only if there is an unmatched '('.",
   "frames": [
    {
     "title": "Start",
     "description": "Empty string.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "( ( ) )",
     "description": "'(' → '((' → '(()' → '(())' complete.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "(",
        ")",
        ")"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "found",
       "cells": [
        "(())"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "open 0 < 2 → add '(' → \"(\"",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "("
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "·"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "open 1 < 2 → add '(' → \"((\"",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          "("
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "·"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "open is 2 = n; close 0 < 2 → add ')' → \"(()\"",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          "(",
          ")"
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "·"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "close 1 < 2 → add ')' → \"(())\"",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          "(",
          ")",
          ")"
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "·"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "length 4 = 2n → save \"(())\"",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          "(",
          ")",
          ")"
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "(())"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Back up: ( ) ( )",
     "description": "Back at '(' choose ')' → '()' → '()(' → '()()' complete.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        ")",
        "(",
        ")"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "found",
       "cells": [
        "(())",
        "()()"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "undo back to \"(\"",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "("
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "(())"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "close 0 < open 1 → add ')' → \"()\"",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          ")"
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "(())"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "open 1 < 2 → add '(' → \"()(\"",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          ")",
          "("
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "(())"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "close 1 < open 2 → add ')' → \"()()\"",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          ")",
          "(",
          ")"
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "(())"
         ],
         "hot": [],
         "marks": {}
        }
       ]
      },
      {
       "description": "length 4 = 2n → save \"()()\"",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "s",
         "cells": [
          "(",
          ")",
          "(",
          ")"
         ],
         "hot": [],
         "marks": {},
         "kind": "list"
        },
        {
         "name": "saved",
         "cells": [
          "(())",
          "()()"
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
     "description": "Both valid strings.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "(())",
        "()()"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
