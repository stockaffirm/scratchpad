window.PROBLEM = {
 "id": "300",
 "title": "Longest Increasing Subsequence",
 "level": "Medium",
 "topic": "1D DP",
 "question": "Find the length of the longest strictly increasing subsequence (numbers in order, not necessarily next to each other).",
 "why": "[10,9,2,5,3,7,101,18] → 2,3,7,101 → length 4.",
 "idea": "Keep a list 'tails' where tails[k] is the smallest possible end of an increasing subsequence of length k+1. Replace using binary search.",
 "input": "[10,9,2,5,3,7,101,18]",
 "output": "4",
 "notes": [],
 "best": "Keep a list 'tails' where tails[k] is the smallest possible end of an increasing subsequence of length k+1. Replace using binary search.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def lengthOfLIS(self, nums):\n        tails = []\n        for x in nums:\n            i = bisect_left(tails, x)\n            if i == len(tails): tails.append(x)\n            else: tails[i] = x\n        return len(tails)",
   "time": "O(n log n)",
   "space": "O(n)",
   "explain": "Keep a list 'tails' where tails[k] is the smallest possible end of an increasing subsequence of length k+1. Replace using binary search.",
   "frames": [
    {
     "title": "Start",
     "description": "tails is empty.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 10",
     "description": "Bigger than all → extend (length 1).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "10"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 9",
     "description": "Replace slot 0 with a smaller end.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "9"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Replace slot 0 with a smaller end.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 5",
     "description": "Bigger than all → extend (length 2).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "2",
        "5"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 3",
     "description": "Replace slot 1 with a smaller end.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "2",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 7",
     "description": "Bigger than all → extend (length 3).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "2",
        "3",
        "7"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 101",
     "description": "Bigger than all → extend (length 4).",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        6
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "2",
        "3",
        "7",
        "101"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "x = 18",
     "description": "Replace slot 3 with a smaller end.",
     "codeLines": [
      5,
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "10",
        "9",
        "2",
        "5",
        "3",
        "7",
        "101",
        "18"
       ],
       "hot": [
        7
       ],
       "marks": {}
      },
      {
       "name": "tails",
       "cells": [
        "2",
        "3",
        "7",
        "18"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "len(tails) = 4",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "4"
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
