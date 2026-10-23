window.PROBLEM = {
 "id": "169",
 "title": "Majority Element",
 "level": "Easy",
 "topic": "Array / String",
 "question": "One number appears more than half the time. Find it.",
 "why": "[2,2,1,1,1,2,2]: 2 appears 4 of 7 times → 2.",
 "idea": "Boyer–Moore voting: keep a candidate and a count. Equal numbers add, different ones cancel out.",
 "input": "[2,2,1,1,1,2,2]",
 "output": "2",
 "notes": [],
 "best": "Boyer–Moore voting: keep a candidate and a count. Equal numbers add, different ones cancel out.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def majorityElement(self, nums):\n        cand, count = None, 0\n        for x in nums:\n            if count == 0: cand = x\n            count += 1 if x == cand else -1\n        return cand",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Boyer–Moore voting: keep a candidate and a count. Equal numbers add, different ones cancel out.",
   "frames": [
    {
     "title": "Start",
     "description": "No candidate yet.",
     "codeLines": [
      3
     ],
     "facts": {
      "cand": "None",
      "count": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
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
     "title": "x = 2",
     "description": "Same as candidate → count up. Candidate 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 2,
      "count": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "x"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Same as candidate → count up. Candidate 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 2,
      "count": 2
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "x"
       }
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Different → cancel one vote. Candidate 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 2,
      "count": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "x"
       }
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Different → cancel one vote. Candidate 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 2,
      "count": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "x"
       }
      }
     ]
    },
    {
     "title": "x = 1",
     "description": "Same as candidate → count up. Candidate 1.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 1,
      "count": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        4
       ],
       "marks": {
        "4": "x"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Different → cancel one vote. Candidate 1.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 1,
      "count": 0
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        5
       ],
       "marks": {
        "5": "x"
       }
      }
     ]
    },
    {
     "title": "x = 2",
     "description": "Same as candidate → count up. Candidate 2.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "cand": 2,
      "count": 1
     },
     "rows": [
      {
       "name": "nums",
       "cells": [
        "2",
        "2",
        "1",
        "1",
        "1",
        "2",
        "2"
       ],
       "hot": [
        6
       ],
       "marks": {
        "6": "x"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "The majority survives all cancelling.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "2"
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
