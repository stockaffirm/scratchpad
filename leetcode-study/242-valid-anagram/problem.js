window.PROBLEM = {
 "id": "242",
 "title": "Valid Anagram",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Are the two words made of exactly the same letters (rearranged)?",
 "why": "'anagram' and 'nagaram' both have a×3, n×1, g×1, r×1, m×1 → True.",
 "idea": "Count the letters in each and compare.",
 "input": "s=\"anagram\", t=\"nagaram\"",
 "output": "True",
 "notes": [],
 "best": "Count the letters in each and compare.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isAnagram(self, s, t):\n        return Counter(s) == Counter(t)",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Count the letters in each and compare.",
   "frames": [
    {
     "title": "Count s",
     "description": "a:3 n:1 g:1 r:1 m:1",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "n",
        "a",
        "g",
        "r",
        "a",
        "m"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "counts",
       "cells": [
        "a:3",
        "n:1",
        "g:1",
        "r:1",
        "m:1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Count t",
     "description": "a:3 n:1 g:1 r:1 m:1",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "t",
       "cells": [
        "n",
        "a",
        "g",
        "a",
        "r",
        "a",
        "m"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "counts",
       "cells": [
        "a:3",
        "n:1",
        "g:1",
        "r:1",
        "m:1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Compare",
     "description": "Equal → True.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
