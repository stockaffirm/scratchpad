window.PROBLEM = {
 "id": "1456",
 "title": "Maximum Vowels in a Substring of Given Length",
 "level": "Medium",
 "topic": "Sliding Window",
 "question": "Most vowels in any substring of length k?",
 "input": "\"abciiidef\", k=3",
 "output": "3",
 "notes": [
  "Booleans add as 0/1 in Python.",
  "Same shape as every fixed-size window problem."
 ],
 "best": "Fixed window: adjust the count by what enters and leaves.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def maxVowels(self, s, k):\n        vowels = set('aeiou')\n        cnt = sum(c in vowels for c in s[:k])\n        best = cnt\n        for i in range(k, len(s)):\n            cnt += (s[i] in vowels) - (s[i-k] in vowels)\n            best = max(best, cnt)\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Fixed-size window; only the entering and leaving letters matter.",
   "frames": [
    {
     "title": "First window",
     "description": "'abc' has 1 vowel.",
     "codeLines": [
      4
     ],
     "facts": {
      "cnt": 1,
      "best": 1
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
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
     "title": "Slide to i=3",
     "description": "In: 'i', out: 'a'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {
      "cnt": 1,
      "best": 1
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
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
     "title": "Slide to i=4",
     "description": "In: 'i', out: 'b'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {
      "cnt": 2,
      "best": 2
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
       ],
       "hot": [
        2,
        3,
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slide to i=5",
     "description": "In: 'i', out: 'c'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {
      "cnt": 3,
      "best": 3
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
       ],
       "hot": [
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slide to i=6",
     "description": "In: 'd', out: 'i'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {
      "cnt": 2,
      "best": 3
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
       ],
       "hot": [
        4,
        5,
        6
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slide to i=7",
     "description": "In: 'e', out: 'i'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {
      "cnt": 2,
      "best": 3
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
       ],
       "hot": [
        5,
        6,
        7
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Slide to i=8",
     "description": "In: 'f', out: 'i'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {
      "cnt": 1,
      "best": 3
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
       ],
       "hot": [
        6,
        7,
        8
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Best window had 3 vowels.",
     "codeLines": [
      9
     ],
     "facts": {
      "best": 3
     },
     "rows": [
      {
       "name": "s",
       "cells": [
        "a",
        "b",
        "c",
        "i",
        "i",
        "i",
        "d",
        "e",
        "f"
       ],
       "hot": [
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
