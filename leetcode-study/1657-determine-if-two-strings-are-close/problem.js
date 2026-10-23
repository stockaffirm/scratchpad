window.PROBLEM = {
 "id": "1657",
 "title": "Determine if Two Strings Are Close",
 "level": "Medium",
 "topic": "Hash Map / Set",
 "question": "Can you turn one string into the other by swapping letters or swapping letter identities?",
 "input": "\"cabbba\", \"abbccc\"",
 "output": "True",
 "notes": [
  "Swapping positions = order is free.",
  "Swapping identities = only the frequency multiset matters."
 ],
 "best": "Two checks: same letter set, same sorted frequency list.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def closeStrings(self, w1, w2):\n        c1, c2 = Counter(w1), Counter(w2)\n        return set(c1) == set(c2) and sorted(c1.values()) == sorted(c2.values())",
   "time": "O(n log n)",
   "space": "O(1)",
   "explain": "Same letters, and same multiset of frequencies.",
   "frames": [
    {
     "title": "Input",
     "description": "Two words.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "w1",
       "cells": [
        "c",
        "a",
        "b",
        "b",
        "b",
        "a"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "w2",
       "cells": [
        "a",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Counts",
     "description": "w1: a2 b3 c1 · w2: a1 b2 c3",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "w1 letters",
       "cells": [
        "a:2",
        "b:3",
        "c:1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "w2 letters",
       "cells": [
        "a:1",
        "b:2",
        "c:3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Same letters?",
     "description": "Both use {a, b, c}.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "set(c1)",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "set(c2)",
       "cells": [
        "a",
        "b",
        "c"
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
     "title": "Same frequencies?",
     "description": "Sorted: [1,2,3] vs [1,2,3].",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "sorted c1",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "sorted c2",
       "cells": [
        "1",
        "2",
        "3"
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
     "title": "Return",
     "description": "Both checks pass.",
     "codeLines": [
      4
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
