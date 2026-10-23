window.PROBLEM = {
 "id": "1431",
 "title": "Kids With the Greatest Number of Candies",
 "level": "Easy",
 "topic": "Arrays",
 "question": "Each kid has some candies. Pretend you give ONE kid all the extra candies. Would that kid then have at least as many as the richest kid? Answer for every kid.",
 "input": "candies=[2,3,5,1,3], extra=3",
 "output": "[T, T, T, F, T]",
 "notes": [
  "max(candies) requires O(n) time even though it is one Python call.",
  "To implement max yourself, start with candies[0] rather than zero (negative numbers!).",
  "Two consecutive O(n) scans still total O(n), not O(n²).",
  "Comparison inside append produces a Boolean (True/False)."
 ],
 "best": "Built-in max() is shorter and equally efficient. Manual max() is useful for learning.",
 "variants": [
  {
   "name": "Built-in max()",
   "code": "class Solution:\n    def kidsWithCandies(self, candies, extraCandies):\n        maxCandies = max(candies)\n        result = []\n        for i in range(len(candies)):\n            result.append(candies[i] + extraCandies >= maxCandies)\n        return result",
   "time": "O(n)",
   "space": "O(n) output; O(1) auxiliary",
   "explain": "Most readable for normal Python.",
   "frames": [
    {
     "title": "Start",
     "description": "Each child hypothetically receives all 3 extra candies.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "candies",
       "cells": [
        "2",
        "3",
        "5",
        "1",
        "3"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Find maximum",
     "description": "max([2,3,5,1,3]) equals 5.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "candies",
       "cells": [
        "2",
        "3",
        "5",
        "1",
        "3"
       ],
       "hot": [
        2
       ]
      }
     ]
    },
    {
     "title": "Check totals",
     "description": "Add 3 to each value independently.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "totals",
       "cells": [
        "5",
        "6",
        "8",
        "4",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        4
       ]
      }
     ],
     "sub": [
      {
       "description": "Kid 0: 2 + 3 = 5 ≥ 5 → True",
       "codeLines": [
        6
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 1: 3 + 3 = 6 ≥ 5 → True",
       "codeLines": [
        6
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "·",
          "·",
          "·"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 2: 5 + 3 = 8 ≥ 5 → True",
       "codeLines": [
        6
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "8",
          "·",
          "·"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 3: 1 + 3 = 4 < 5 → False",
       "codeLines": [
        6
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          3
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "8",
          "4",
          "·"
         ],
         "hot": [
          3
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 4: 3 + 3 = 6 ≥ 5 → True",
       "codeLines": [
        6
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          4
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "8",
          "4",
          "6"
         ],
         "hot": [
          4
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Return Booleans",
     "description": "True means the child can reach at least the original maximum.",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "T",
        "T",
        "T",
        "F",
        "T"
       ],
       "hot": [
        0,
        1,
        2,
        4
       ]
      }
     ]
    }
   ]
  },
  {
   "name": "Manual max",
   "code": "class Solution:\n    def kidsWithCandies(self, candies, extraCandies):\n        biggest = candies[0]\n        for value in candies:\n            if value > biggest: biggest = value\n        result = []\n        for value in candies:\n            result.append(value + extraCandies >= biggest)\n        return result",
   "time": "O(n)",
   "space": "O(n) output; O(1) auxiliary",
   "explain": "Shows the logic inside Python's max() helper.",
   "frames": [
    {
     "title": "Start with first item",
     "description": "biggest = candies[0] = 2.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "candies",
       "cells": [
        "2",
        "3",
        "5",
        "1",
        "3"
       ],
       "hot": [
        0
       ]
      }
     ]
    },
    {
     "title": "Scan and update",
     "description": "Observe 3, then 5; running maximum becomes 5.",
     "codeLines": [
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "running max",
       "cells": [
        "2",
        "3",
        "5",
        "5",
        "5"
       ],
       "hot": [
        2
       ]
      }
     ],
     "sub": [
      {
       "description": "Start: biggest = 2",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "running max",
         "cells": [
          "2",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "3 vs biggest 2 → new biggest 3",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "running max",
         "cells": [
          "2",
          "3",
          "·",
          "·",
          "·"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "5 vs biggest 3 → new biggest 5",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "running max",
         "cells": [
          "2",
          "3",
          "5",
          "·",
          "·"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "1 vs biggest 5 → keep 5",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          3
         ],
         "marks": {}
        },
        {
         "name": "running max",
         "cells": [
          "2",
          "3",
          "5",
          "5",
          "·"
         ],
         "hot": [
          3
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "3 vs biggest 5 → keep 5",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          4
         ],
         "marks": {}
        },
        {
         "name": "running max",
         "cells": [
          "2",
          "3",
          "5",
          "5",
          "5"
         ],
         "hot": [
          4
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Check each child",
     "description": "Compare candy + 3 >= 5.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "totals",
       "cells": [
        "5",
        "6",
        "8",
        "4",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        4
       ]
      }
     ],
     "sub": [
      {
       "description": "Kid 0: 2 + 3 = 5 ≥ 5 → True",
       "codeLines": [
        8
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "·",
          "·",
          "·",
          "·"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 1: 3 + 3 = 6 ≥ 5 → True",
       "codeLines": [
        8
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "·",
          "·",
          "·"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 2: 5 + 3 = 8 ≥ 5 → True",
       "codeLines": [
        8
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "8",
          "·",
          "·"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 3: 1 + 3 = 4 < 5 → False",
       "codeLines": [
        8
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          3
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "8",
          "4",
          "·"
         ],
         "hot": [
          3
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "Kid 4: 3 + 3 = 6 ≥ 5 → True",
       "codeLines": [
        8
       ],
       "facts": {
        "max": 5
       },
       "rows": [
        {
         "name": "candies",
         "cells": [
          "2",
          "3",
          "5",
          "1",
          "3"
         ],
         "hot": [
          4
         ],
         "marks": {}
        },
        {
         "name": "with extra",
         "cells": [
          "5",
          "6",
          "8",
          "4",
          "6"
         ],
         "hot": [
          4
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "Same Boolean answer as max().",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "T",
        "T",
        "T",
        "F",
        "T"
       ],
       "hot": []
      }
     ]
    }
   ]
  }
 ],
 "why": "Most candies is 5, extra is 3. Kid with 2 → 5 ✓, 3 → 6 ✓, 5 → 8 ✓, 1 → 4 ✗, 3 → 6 ✓.",
 "idea": "Find the maximum once. Then check candies[i] + extra ≥ max."
};
