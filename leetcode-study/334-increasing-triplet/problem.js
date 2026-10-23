window.PROBLEM = {
 "id": "334",
 "title": "Increasing Triplet Subsequence",
 "level": "Medium",
 "topic": "Greedy",
 "question": "Is there i < j < k with nums[i] < nums[j] < nums[k]?",
 "input": "[2, 1, 5, 0, 4, 6]",
 "output": "True",
 "notes": [
  "Indices must increase and values must be strictly increasing, but positions need not be adjacent.",
  "first_num tracks a small first threshold; second_num tracks a small possible middle threshold.",
  "<= is deliberate: duplicates must not count as a strictly increasing triplet.",
  "[2,1,5,0,4,6] has valid triplets (2,5,6), (1,5,6), and (0,4,6).",
  "The task only asks whether a triplet exists; it does not need to return which one."
 ],
 "best": "Greedy keeps improving small thresholds, making a future larger third value easier to find.",
 "variants": [
  {
   "name": "Greedy two thresholds",
   "code": "class Solution:\n    def increasingTriplet(self, nums):\n        first_num = float('inf')\n        second_num = float('inf')\n        for n in nums:\n            if n <= first_num: first_num = n\n            elif n <= second_num: second_num = n\n            else: return True\n        return False",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Greedy keeps improving small thresholds, making a future larger third value easier to find.",
   "frames": [
    {
     "title": "Input",
     "description": "We need indices i<j<k and nums[i]<nums[j]<nums[k].",
     "codeLines": [
      2,
      3,
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "1",
        "5",
        "0",
        "4",
        "6"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "See 2 and 1",
     "description": "Update first from ∞→2→1.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "first": 1,
      "second": "∞"
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "1",
        "5",
        "0",
        "4",
        "6"
       ],
       "hot": [
        0,
        1
       ]
      }
     ]
    },
    {
     "title": "See 5",
     "description": "5 is greater than first=1; set second=5.",
     "codeLines": [
      7
     ],
     "facts": {
      "first": 1,
      "second": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "1",
        "5",
        "0",
        "4",
        "6"
       ],
       "hot": [
        2
       ]
      }
     ]
    },
    {
     "title": "See 0",
     "description": "Update first=0, while second remains 5.",
     "codeLines": [
      6
     ],
     "facts": {
      "first": 0,
      "second": 5
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "1",
        "5",
        "0",
        "4",
        "6"
       ],
       "hot": [
        3
       ]
      }
     ]
    },
    {
     "title": "See 4",
     "description": "4 is smaller than second=5, so second becomes 4.",
     "codeLines": [
      7
     ],
     "facts": {
      "first": 0,
      "second": 4
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "1",
        "5",
        "0",
        "4",
        "6"
       ],
       "hot": [
        3,
        4
       ]
      }
     ]
    },
    {
     "title": "See 6: success",
     "description": "6 exceeds both thresholds: triplet 0<4<6. Also 2<5<6 is valid.",
     "codeLines": [
      8
     ],
     "facts": {
      "result": "True"
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "1",
        "5",
        "0",
        "4",
        "6"
       ],
       "hot": [
        3,
        4,
        5
       ]
      }
     ]
    }
   ]
  }
 ]
};
