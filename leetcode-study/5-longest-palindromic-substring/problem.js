window.PROBLEM = {
 "id": "5",
 "title": "Longest Palindromic Substring",
 "level": "Medium",
 "topic": "Multidimensional DP",
 "question": "Find the longest piece of the text that reads the same backwards.",
 "why": "'babad' → 'bab' (or 'aba') → length 3.",
 "idea": "Treat every position (and every gap) as a center and expand outwards while the letters match.",
 "input": "\"babad\"",
 "output": "\"bab\"",
 "notes": [],
 "best": "Treat every position (and every gap) as a center and expand outwards while the letters match.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestPalindrome(self, s):\n        best = ''\n        for c in range(len(s)):\n            for l, r in ((c, c), (c, c + 1)):\n                while l >= 0 and r < len(s) and s[l] == s[r]: l -= 1; r += 1\n                if r - l - 1 > len(best): best = s[l+1:r]\n        return best",
   "time": "O(n²)",
   "space": "O(1)",
   "explain": "Treat every position (and every gap) as a center and expand outwards while the letters match.",
   "frames": [
    {
     "title": "Start",
     "description": "Try each letter as a center.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "b",
        "a",
        "d"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Center 'b' (0)",
     "description": "Can't expand left → 'b'.",
     "codeLines": [
      6
     ],
     "facts": {
      "best": "b"
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "b",
        "a",
        "d"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Center 'a' (1)",
     "description": "Expand: b = b → 'bab'.",
     "codeLines": [
      6,
      7
     ],
     "facts": {
      "best": "bab"
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "b",
        "a",
        "d"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Other centers",
     "description": "Centers at 2, 3, 4 give 'aba' (length 3, not longer), 'd', etc.",
     "codeLines": [
      7
     ],
     "facts": {
      "best": "bab"
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "b",
        "a",
        "d"
       ],
       "hot": [
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "'bab'",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "b",
        "a",
        "b"
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
