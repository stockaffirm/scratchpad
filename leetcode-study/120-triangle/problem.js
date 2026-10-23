window.PROBLEM = {
 "id": "120",
 "title": "Triangle",
 "level": "Medium",
 "topic": "Multidimensional DP",
 "question": "Walk from the top of a number triangle to the bottom, moving to an adjacent number each row. What's the smallest path sum?",
 "why": "[[2],[3,4],[6,5,7],[4,1,8,3]] → 2 + 3 + 5 + 1 = 11.",
 "idea": "Work from the bottom up: each cell = its value + the smaller of the two cells below it.",
 "input": "[[2],[3,4],[6,5,7],[4,1,8,3]]",
 "output": "11",
 "notes": [],
 "best": "Work from the bottom up: each cell = its value + the smaller of the two cells below it.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def minimumTotal(self, triangle):\n        dp = triangle[-1][:]\n        for row in range(len(triangle) - 2, -1, -1):\n            for i in range(len(triangle[row])):\n                dp[i] = triangle[row][i] + min(dp[i], dp[i + 1])\n        return dp[0]",
   "time": "O(n²)",
   "space": "O(n)",
   "explain": "Work from the bottom up: each cell = its value + the smaller of the two cells below it.",
   "frames": [
    {
     "title": "Bottom row",
     "description": "Start with the last row: [4,1,8,3].",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "4",
        "1",
        "8",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Row [6,5,7]",
     "description": "6+min(4,1)=7, 5+min(1,8)=6, 7+min(8,3)=10",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "7",
        "6",
        "10"
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
       "description": "dp[0] = 6 + min(4, 1) = 7",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "row",
         "cells": [
          "6",
          "5",
          "7"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "dp",
         "cells": [
          "7",
          "1",
          "8"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "dp[1] = 5 + min(1, 8) = 6",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "row",
         "cells": [
          "6",
          "5",
          "7"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "dp",
         "cells": [
          "7",
          "6",
          "8"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "dp[2] = 7 + min(8, 3) = 10",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "row",
         "cells": [
          "6",
          "5",
          "7"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "dp",
         "cells": [
          "7",
          "6",
          "10"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Row [3,4]",
     "description": "3+min(7,6)=9, 4+min(6,10)=10",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "9",
        "10"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "dp[0] = 3 + min(7, 6) = 9",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "row",
         "cells": [
          "3",
          "4"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "dp",
         "cells": [
          "9",
          "6"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "dp[1] = 4 + min(6, 10) = 10",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "row",
         "cells": [
          "3",
          "4"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "dp",
         "cells": [
          "9",
          "10"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Top",
     "description": "2 + min(9,10) = 11",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "dp",
       "cells": [
        "11"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "dp[0] = 2 + min(9, 10) = 11",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "row",
         "cells": [
          "2"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "dp",
         "cells": [
          "11"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "11",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "11"
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
