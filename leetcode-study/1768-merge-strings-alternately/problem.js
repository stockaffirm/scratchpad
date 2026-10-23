window.PROBLEM = {
 "id": "1768",
 "title": "Merge Strings Alternately",
 "level": "Easy",
 "topic": "Array / String",
 "question": "Merge two strings by taking letters alternately; append any leftover letters.",
 "input": "\"abc\", \"pq\"",
 "output": "\"apbqc\"",
 "notes": [
  "Loop to max(len1, len2); guard each append with i < len.",
  "Building a list and joining once beats repeated string +=."
 ],
 "best": "One pass with index guards handles unequal lengths with no special case.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def mergeAlternately(self, word1, word2):\n        res = []\n        for i in range(max(len(word1), len(word2))):\n            if i < len(word1): res.append(word1[i])\n            if i < len(word2): res.append(word2[i])\n        return ''.join(res)",
   "time": "O(n+m)",
   "space": "O(n+m)",
   "explain": "One loop up to the longer length, guarding each side.",
   "frames": [
    {
     "title": "Input",
     "description": "Two words, different lengths.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "word1",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "word2",
       "cells": [
        "p",
        "q"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 0",
     "description": "Append word1[0] then word2[0].",
     "codeLines": [
      4,
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "word1",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "word2",
       "cells": [
        "p",
        "q"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "res",
       "cells": [
        "a",
        "p"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 1",
     "description": "Append word1[1] then word2[1].",
     "codeLines": [
      4,
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "word1",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "word2",
       "cells": [
        "p",
        "q"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "res",
       "cells": [
        "a",
        "p",
        "b",
        "q"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 2",
     "description": "Append word1[2].",
     "codeLines": [
      4,
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "word1",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "word2",
       "cells": [
        "p",
        "q"
       ],
       "hot": [],
       "marks": {
        "2": "i"
       }
      },
      {
       "name": "res",
       "cells": [
        "a",
        "p",
        "b",
        "q",
        "c"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Join",
     "description": "Join the list into the answer.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "a",
        "p",
        "b",
        "q",
        "c"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
