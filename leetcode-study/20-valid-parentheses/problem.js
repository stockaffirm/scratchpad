window.PROBLEM = {
 "id": "20",
 "title": "Valid Parentheses",
 "level": "Easy",
 "topic": "Stack",
 "question": "Is every bracket ( [ { closed by the matching bracket in the right order?",
 "why": "'([{}])' is perfectly nested → True. '(]' would be False.",
 "idea": "Push opening brackets. On a closing bracket, the top of the stack must be its partner.",
 "input": "\"([{}])\"",
 "output": "True",
 "notes": [],
 "best": "Push opening brackets. On a closing bracket, the top of the stack must be its partner.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isValid(self, s):\n        stack, pairs = [], {')':'(', ']':'[', '}':'{'}\n        for ch in s:\n            if ch in pairs:\n                if not stack or stack.pop() != pairs[ch]: return False\n            else: stack.append(ch)\n        return not stack",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Push opening brackets. On a closing bracket, the top of the stack must be its partner.",
   "frames": [
    {
     "title": "Start",
     "description": "Empty stack.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'('",
     "description": "Opening → push.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "("
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'['",
     "description": "Opening → push.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "(",
        "["
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'{'",
     "description": "Opening → push.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "(",
        "[",
        "{"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'}'",
     "description": "Closing → pops '{' (its partner).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "(",
        "["
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "']'",
     "description": "Closing → pops '[' (its partner).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "("
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "')'",
     "description": "Closing → pops '(' (its partner).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "(",
        "[",
        "{",
        "}",
        "]",
        ")"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Stack empty → True.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
