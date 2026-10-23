window.PROBLEM = {
 "id": "2215",
 "title": "Find the Difference of Two Arrays",
 "level": "Easy",
 "topic": "Hash Map / Set",
 "question": "Values only in nums1, and values only in nums2 (no duplicates).",
 "input": "[1,2,3,3], [1,1,2,2]",
 "output": "[[3], []]",
 "notes": [
  "set(x) removes duplicates in O(n).",
  "a - b keeps what a has and b lacks."
 ],
 "best": "Two sets and set difference.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findDifference(self, nums1, nums2):\n        a, b = set(nums1), set(nums2)\n        return [list(a - b), list(b - a)]",
   "time": "O(n+m)",
   "space": "O(n+m)",
   "explain": "Sets remove duplicates; subtraction gives differences.",
   "frames": [
    {
     "title": "Input",
     "description": "Two arrays with duplicates.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "3",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "nums2",
       "cells": [
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "To sets",
     "description": "Duplicates vanish.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "a − b",
     "description": "Only 3 is in a but not b.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "a − b",
       "cells": [
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "b − a",
     "description": "Nothing in b is missing from a.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "a",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "b",
       "cells": [
        "1",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "b − a",
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
     "description": "[[3], []]",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "[3]",
        "[]"
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
