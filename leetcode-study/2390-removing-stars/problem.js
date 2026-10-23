window.PROBLEM = {
 "id": "2390",
 "title": "Removing Stars From a String",
 "level": "Medium",
 "topic": "Stack",
 "question": "Each * deletes the closest non-star letter to its left. Return what remains.",
 "input": "\"leet**cod*e\"",
 "output": "\"lecoe\"",
 "notes": [
  "'Closest to the left' = last in, first out.",
  "Join the stack once at the end."
 ],
 "best": "A stack: the closest letter to the left is always the top.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def removeStars(self, s):\n        stack = []\n        for ch in s:\n            if ch == '*': stack.pop()\n            else: stack.append(ch)\n        return ''.join(stack)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "A star pops the latest letter.",
   "frames": [
    {
     "title": "Input",
     "description": "Scan left to right with a stack.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
     "title": "'l'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'e'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'e'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e",
        "e"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'t'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e",
        "e",
        "t"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'*'",
     "description": "Pop 't'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e",
        "e"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'*'",
     "description": "Pop 'e'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'c'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e",
        "c"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'o'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
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
        "l",
        "e",
        "c",
        "o"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'d'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
       ],
       "hot": [
        8
       ],
       "marks": {
        "8": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "l",
        "e",
        "c",
        "o",
        "d"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'*'",
     "description": "Pop 'd'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
       ],
       "hot": [
        9
       ],
       "marks": {
        "9": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "l",
        "e",
        "c",
        "o"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'e'",
     "description": "Push.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "l",
        "e",
        "e",
        "t",
        "*",
        "*",
        "c",
        "o",
        "d",
        "*",
        "e"
       ],
       "hot": [
        10
       ],
       "marks": {
        "10": "i"
       }
      },
      {
       "name": "stack",
       "cells": [
        "l",
        "e",
        "c",
        "o",
        "e"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Join the stack.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "l",
        "e",
        "c",
        "o",
        "e"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
