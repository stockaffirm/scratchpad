window.PROBLEM = {
 "id": "88",
 "title": "Merge Sorted Array",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Two sorted lists. nums1 has empty room (zeros) at the end. Merge nums2 into nums1 so the whole thing is sorted, in place.",
 "why": "nums1 = [1,2,3,_,_,_], nums2 = [2,5,6] → [1,2,2,3,5,6].",
 "idea": "Fill nums1 from the BACK with the biggest remaining value, so nothing gets overwritten.",
 "input": "nums1=[1,2,3,0,0,0], m=3, nums2=[2,5,6], n=3",
 "output": "[1,2,2,3,5,6]",
 "notes": [],
 "best": "Fill nums1 from the BACK with the biggest remaining value, so nothing gets overwritten.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def merge(self, nums1, m, nums2, n):\n        i, j, k = m - 1, n - 1, m + n - 1\n        while j >= 0:\n            if i >= 0 and nums1[i] > nums2[j]:\n                nums1[k] = nums1[i]; i -= 1\n            else:\n                nums1[k] = nums2[j]; j -= 1\n            k -= 1",
   "time": "O(m+n)",
   "space": "O(1)",
   "explain": "Fill nums1 from the BACK with the biggest remaining value, so nothing gets overwritten.",
   "frames": [
    {
     "title": "Start",
     "description": "Pointers at the ends: i (nums1 data), j (nums2), k (write spot).",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "3",
        "0",
        "0",
        "0"
       ],
       "hot": [],
       "marks": {
        "2": "i",
        "5": "k"
       }
      },
      {
       "name": "nums2",
       "cells": [
        "2",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {
        "2": "j"
       }
      }
     ]
    },
    {
     "title": "Place at 5",
     "description": "6 (nums2) wins → write it at k.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "3",
        "0",
        "0",
        "6"
       ],
       "hot": [
        5
       ],
       "marks": {
        "2": "i",
        "4": "k"
       }
      },
      {
       "name": "nums2",
       "cells": [
        "2",
        "5",
        "6"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "j"
       }
      }
     ]
    },
    {
     "title": "Place at 4",
     "description": "5 (nums2) wins → write it at k.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "3",
        "0",
        "5",
        "6"
       ],
       "hot": [
        4
       ],
       "marks": {
        "2": "i",
        "3": "k"
       }
      },
      {
       "name": "nums2",
       "cells": [
        "2",
        "5",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "j"
       }
      }
     ]
    },
    {
     "title": "Place at 3",
     "description": "3 (nums1) is bigger → write it at k.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "3",
        "3",
        "5",
        "6"
       ],
       "hot": [
        3
       ],
       "marks": {
        "1": "i",
        "2": "k"
       }
      },
      {
       "name": "nums2",
       "cells": [
        "2",
        "5",
        "6"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "j"
       }
      }
     ]
    },
    {
     "title": "Place at 2",
     "description": "2 (nums2) wins → write it at k.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "2",
        "3",
        "5",
        "6"
       ],
       "hot": [
        2
       ],
       "marks": {
        "1": "i,k"
       }
      },
      {
       "name": "nums2",
       "cells": [
        "2",
        "5",
        "6"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Done",
     "description": "nums2 is used up; nums1 is sorted.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "nums1",
       "cells": [
        "1",
        "2",
        "2",
        "3",
        "5",
        "6"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
