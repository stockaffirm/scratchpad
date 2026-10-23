window.PROBLEM = {
 "id": "219",
 "title": "Contains Duplicate II",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Are there two equal numbers whose positions differ by at most k?",
 "why": "[1,2,3,1], k = 3: the two 1s are 3 apart → True.",
 "idea": "Remember each number's last position in a dictionary.",
 "input": "[1,2,3,1], k=3",
 "output": "True",
 "notes": [],
 "best": "Remember each number's last position in a dictionary.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def containsNearbyDuplicate(self, nums, k):\n        last = {}\n        for i, x in enumerate(nums):\n            if x in last and i - last[x] <= k: return True\n            last[x] = i\n        return False",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Remember each number's last position in a dictionary.",
   "frames": [
    {
     "title": "Start",
     "description": "Track the last index of each value.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "3",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "1, 2, 3",
     "description": "All new → record positions.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "3",
        "1"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "last",
       "cells": [
        "1→0",
        "2→1",
        "3→2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Second 1",
     "description": "Last seen at 0; distance 3 − 0 = 3 ≤ k → True.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums",
       "cells": [
        "1",
        "2",
        "3",
        "1"
       ],
       "hot": [
        0,
        3
       ],
       "marks": {}
      },
      {
       "name": "last",
       "cells": [
        "1→0",
        "2→1",
        "3→2"
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
