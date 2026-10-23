window.PROBLEM = {
 "id": "373",
 "title": "Find K Pairs with Smallest Sums",
 "level": "Medium",
 "topic": "Heap",
 "question": "Two sorted lists. Pick one number from each to make a pair. List the k pairs with the smallest sums.",
 "why": "[1,7,11] and [2,4,6], k = 3 → (1,2), (1,4), (1,6).",
 "idea": "A min-heap of candidate pairs; after taking (i, j), push the next pair (i, j+1).",
 "input": "nums1=[1,7,11], nums2=[2,4,6], k=3",
 "output": "[[1,2],[1,4],[1,6]]",
 "notes": [],
 "best": "A min-heap of candidate pairs; after taking (i, j), push the next pair (i, j+1).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def kSmallestPairs(self, nums1, nums2, k):\n        heap = [(nums1[i] + nums2[0], i, 0) for i in range(min(k, len(nums1)))]; heapify(heap)\n        res = []\n        while heap and len(res) < k:\n            s, i, j = heappop(heap)\n            res.append([nums1[i], nums2[j]])\n            if j + 1 < len(nums2): heappush(heap, (nums1[i] + nums2[j+1], i, j + 1))\n        return res",
   "time": "O(k log k)",
   "space": "O(k)",
   "explain": "A min-heap of candidate pairs; after taking (i, j), push the next pair (i, j+1).",
   "frames": [
    {
     "title": "Start",
     "description": "Pair each nums1 value with the first of nums2: sums 3, 9, 13.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (sum)",
       "cells": [
        "3",
        "9",
        "13"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pop (1,2)",
     "description": "Smallest sum 3. Push the next pair for 1: (1,4) = 5.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (sum)",
       "cells": [
        "5",
        "9",
        "13"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "[1,2]"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pop (1,4)",
     "description": "Sum 5. Push (1,6) = 7.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "heap (sum)",
       "cells": [
        "7",
        "9",
        "13"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "result",
       "cells": [
        "[1,2]",
        "[1,4]"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pop (1,6)",
     "description": "Sum 7 → we have k = 3 pairs.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "[1,2]",
        "[1,4]",
        "[1,6]"
       ],
       "hot": [
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
