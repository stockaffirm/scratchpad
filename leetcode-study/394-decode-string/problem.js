window.PROBLEM = {
 "id": "394",
 "title": "Decode String",
 "level": "Medium",
 "topic": "Stack",
 "question": "Expand k[encoded] patterns, possibly nested.",
 "input": "\"3[a2[c]]\"",
 "output": "\"accaccacc\"",
 "notes": [
  "Multi-digit numbers: num = num*10 + digit.",
  "Each ']' repeats cur and glues it to the saved prefix."
 ],
 "best": "Stack saves the outer string and repeat count at each '['.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def decodeString(self, s):\n        stack, cur, num = [], '', 0\n        for ch in s:\n            if ch.isdigit(): num = num * 10 + int(ch)\n            elif ch == '[':\n                stack.append((cur, num)); cur, num = '', 0\n            elif ch == ']':\n                prev, k = stack.pop(); cur = prev + cur * k\n            else: cur += ch\n        return cur",
   "time": "O(output)",
   "space": "O(depth)",
   "explain": "Push state at '[', pop and repeat at ']'.",
   "frames": [
    {
     "title": "Input",
     "description": "Nested repeat counts.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'3'",
     "description": "Digit: num = 3",
     "codeLines": [
      5
     ],
     "facts": {
      "num": 3
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
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
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'['",
     "description": "Save ('', 3) on stack",
     "codeLines": [
      7
     ],
     "facts": {
      "num": 0
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
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
        "(,3)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'a'",
     "description": "Letter: cur = 'a'",
     "codeLines": [
      10
     ],
     "facts": {
      "num": 0
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
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
        "(,3)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "a"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'2'",
     "description": "Digit: num = 2",
     "codeLines": [
      5
     ],
     "facts": {
      "num": 2
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
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
        "(,3)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "a"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'['",
     "description": "Save ('a', 2) on stack",
     "codeLines": [
      7
     ],
     "facts": {
      "num": 0
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
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
        "(,3)",
        "(a,2)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'c'",
     "description": "Letter: cur = 'c'",
     "codeLines": [
      10
     ],
     "facts": {
      "num": 0
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
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
        "(,3)",
        "(a,2)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "']'",
     "description": "Pop → 'a' + 'c'×2 = 'acc'",
     "codeLines": [
      9
     ],
     "facts": {
      "num": 0
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
       ],
       "hot": [
        6
       ],
       "marks": {
        "6": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "(,3)"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "a",
        "c",
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "']'",
     "description": "Pop → '' + 'acc'×3 = 'accaccacc'",
     "codeLines": [
      9
     ],
     "facts": {
      "num": 0
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "3",
        "[",
        "a",
        "2",
        "[",
        "c",
        "]",
        "]"
       ],
       "hot": [
        7
       ],
       "marks": {
        "7": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "cur",
       "cells": [
        "a",
        "c",
        "c",
        "a",
        "c",
        "c",
        "a",
        "c",
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Fully expanded.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "a",
        "c",
        "c",
        "a",
        "c",
        "c",
        "a",
        "c",
        "c"
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
 ]
};
