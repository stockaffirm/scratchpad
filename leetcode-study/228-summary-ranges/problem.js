window.PROBLEM = {
 "id": "228",
 "title": "Summary Ranges",
 "level": "Easy",
 "topic": "Intervals",
 "question": "Turn a sorted list of unique numbers into the shortest list of ranges.",
 "why": "[0,1,2,4,5,7] → 0→2, 4→5, 7.",
 "idea": "Walk along; extend each run while the next number is exactly one bigger.",
 "input": "[0,1,2,4,5,7]",
 "output": "[\"0->2\",\"4->5\",\"7\"]",
 "notes": [],
 "best": "Walk along; extend each run while the next number is exactly one bigger.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def summaryRanges(self, nums):\n        res, i = [], 0\n        while i < len(nums):\n            j = i\n            while j + 1 < len(nums) and nums[j+1] == nums[j] + 1: j += 1\n            res.append(str(nums[i]) if i == j else f'{nums[i]}->{nums[j]}')\n            i = j + 1\n        return res",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Walk along; extend each run while the next number is exactly one bigger.",
   "frames": [
    {
     "title": "Start",
     "description": "Find the end of each run.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "4",
        "5",
        "7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Run 0…2",
     "description": "Record 0->2",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "4",
        "5",
        "7"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {
        "0": "i",
        "2": "j"
       }
      }
     ],
     "sub": [
      {
       "description": "start a run at nums[0] = 0",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          0
         ],
         "marks": {
          "0": "i,j"
         }
        }
       ]
      },
      {
       "description": "1 = 0 + 1 → extend the run (j = 1)",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          0,
          1
         ],
         "marks": {
          "0": "i",
          "1": "j"
         }
        }
       ]
      },
      {
       "description": "2 = 1 + 1 → extend the run (j = 2)",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          0,
          1,
          2
         ],
         "marks": {
          "0": "i",
          "2": "j"
         }
        }
       ]
      },
      {
       "description": "4 ≠ 2 + 1 → the run ends. Record 0->2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          0,
          1,
          2
         ],
         "marks": {
          "0": "i",
          "2": "j"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Run 4…5",
     "description": "Record 4->5",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "4",
        "5",
        "7"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {
        "3": "i",
        "4": "j"
       }
      }
     ],
     "sub": [
      {
       "description": "start a run at nums[3] = 4",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          3
         ],
         "marks": {
          "3": "i,j"
         }
        }
       ]
      },
      {
       "description": "5 = 4 + 1 → extend the run (j = 4)",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          3,
          4
         ],
         "marks": {
          "3": "i",
          "4": "j"
         }
        }
       ]
      },
      {
       "description": "7 ≠ 5 + 1 → the run ends. Record 4->5",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          3,
          4
         ],
         "marks": {
          "3": "i",
          "4": "j"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Run 7…7",
     "description": "Record 7",
     "codeLines": [
      6,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "0",
        "1",
        "2",
        "4",
        "5",
        "7"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "i,j"
       }
      }
     ],
     "sub": [
      {
       "description": "start a run at nums[5] = 7",
       "codeLines": [
        5
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          5
         ],
         "marks": {
          "5": "i,j"
         }
        }
       ]
      },
      {
       "description": "End of list. Record 7",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "nums",
         "cells": [
          "0",
          "1",
          "2",
          "4",
          "5",
          "7"
         ],
         "hot": [
          5
         ],
         "marks": {
          "5": "i,j"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "[\"0->2\",\"4->5\",\"7\"]",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "res",
       "cells": [
        "0->2",
        "4->5",
        "7"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
