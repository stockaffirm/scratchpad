window.PROBLEM = {
 "id": "71",
 "title": "Simplify Path",
 "level": "Medium",
 "topic": "Stack",
 "question": "Clean up a Unix file path: '.' means stay, '..' means go up one folder, and extra slashes collapse.",
 "why": "'/a/./b/../../c/' → into a, stay, into b, up, up, into c → '/c'.",
 "idea": "Split on '/'; push folder names, pop on '..'.",
 "input": "\"/a/./b/../../c/\"",
 "output": "\"/c\"",
 "notes": [],
 "best": "Split on '/'; push folder names, pop on '..'.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def simplifyPath(self, path):\n        stack = []\n        for part in path.split('/'):\n            if part == '..':\n                if stack: stack.pop()\n            elif part and part != '.': stack.append(part)\n        return '/' + '/'.join(stack)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Split on '/'; push folder names, pop on '..'.",
   "frames": [
    {
     "title": "Split",
     "description": "Pieces between slashes.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "a",
     "description": "Folder → push.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "a"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": ".",
     "description": "'.' → stay, ignore.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "a"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "b",
     "description": "Folder → push.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "a",
        "b"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "..",
     "description": "'..' → go up (pop).",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "a"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "..",
     "description": "'..' → go up (pop).",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [
        4
       ],
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
     "title": "c",
     "description": "Folder → push.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "parts",
       "cells": [
        "a",
        ".",
        "b",
        "..",
        "..",
        "c"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "c"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "'/' + 'c'",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "/",
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
