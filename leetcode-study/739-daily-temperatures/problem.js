window.PROBLEM = {
 "id": "739",
 "title": "Daily Temperatures",
 "level": "Medium",
 "topic": "Monotonic Stack",
 "question": "For each day, how many days until a warmer one? (0 if never.)",
 "input": "[73,74,75,71,69,72,76,73]",
 "output": "[1,1,4,2,1,1,0,0]",
 "notes": [
  "A warmer day resolves every colder day on the stack.",
  "Each index is pushed and popped once."
 ],
 "best": "Monotonic decreasing stack of indices.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def dailyTemperatures(self, temps):\n        ans, stack = [0] * len(temps), []\n        for i, t in enumerate(temps):\n            while stack and temps[stack[-1]] < t:\n                j = stack.pop(); ans[j] = i - j\n            stack.append(i)\n        return ans",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Stack of days still waiting for a warmer one.",
   "frames": [
    {
     "title": "Input",
     "description": "Stack holds days waiting for a warmer day.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 0 (73)",
     "description": "Not warmer than the top; wait.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 1 (74)",
     "description": "Warmer than days 0 → resolve them.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 2 (75)",
     "description": "Warmer than days 1 → resolve them.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 3 (71)",
     "description": "Not warmer than the top; wait.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "2",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 4 (69)",
     "description": "Not warmer than the top; wait.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "2",
        "3",
        "4"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 5 (72)",
     "description": "Warmer than days 4, 3 → resolve them.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "2",
        "5"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "0",
        "2",
        "1",
        "0",
        "0",
        "0"
       ],
       "hot": [
        4,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 6 (76)",
     "description": "Warmer than days 5, 2 → resolve them.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        6
       ],
       "marks": {
        "6": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "4",
        "2",
        "1",
        "1",
        "0",
        "0"
       ],
       "hot": [
        5,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Day 7 (73)",
     "description": "Not warmer than the top; wait.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "temps",
       "cells": [
        "73",
        "74",
        "75",
        "71",
        "69",
        "72",
        "76",
        "73"
       ],
       "hot": [
        7
       ],
       "marks": {
        "7": "i"
       }
      },
      {
       "name": "stack (days)",
       "cells": [
        "6",
        "7"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "4",
        "2",
        "1",
        "1",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Days left on the stack never get warmer → 0.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "ans",
       "cells": [
        "1",
        "1",
        "4",
        "2",
        "1",
        "1",
        "0",
        "0"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
