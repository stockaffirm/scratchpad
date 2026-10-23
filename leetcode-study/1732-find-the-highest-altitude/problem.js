window.PROBLEM = {
 "id": "1732",
 "title": "Find the Highest Altitude",
 "level": "Easy",
 "topic": "Prefix Sum",
 "question": "A biker starts at 0 and gains/loses altitude per leg. Highest point?",
 "input": "gain = [-5, 1, 5, 0, -7]",
 "output": "1",
 "notes": [
  "Include the starting altitude 0 in the max.",
  "Prefix sum = running total."
 ],
 "best": "A running prefix sum with a max — no array needed.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def largestAltitude(self, gain):\n        alt = best = 0\n        for g in gain:\n            alt += g\n            best = max(best, alt)\n        return best",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Running sum of the gains; track the max.",
   "frames": [
    {
     "title": "Start",
     "description": "Altitude 0 at the start.",
     "codeLines": [
      3
     ],
     "facts": {
      "alt": 0,
      "best": 0
     },
     "rows": [
      {
       "name": "gain",
       "cells": [
        "-5",
        "1",
        "5",
        "0",
        "-7"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leg 1",
     "description": "alt = -5",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "alt": -5,
      "best": 0
     },
     "rows": [
      {
       "name": "gain",
       "cells": [
        "-5",
        "1",
        "5",
        "0",
        "-7"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "altitude",
       "cells": [
        "0",
        "-5"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leg 2",
     "description": "alt = -4",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "alt": -4,
      "best": 0
     },
     "rows": [
      {
       "name": "gain",
       "cells": [
        "-5",
        "1",
        "5",
        "0",
        "-7"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "altitude",
       "cells": [
        "0",
        "-5",
        "-4"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leg 3",
     "description": "alt = 1 (new best)",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "alt": 1,
      "best": 1
     },
     "rows": [
      {
       "name": "gain",
       "cells": [
        "-5",
        "1",
        "5",
        "0",
        "-7"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "altitude",
       "cells": [
        "0",
        "-5",
        "-4",
        "1"
       ],
       "hot": [
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leg 4",
     "description": "alt = 1 (new best)",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "alt": 1,
      "best": 1
     },
     "rows": [
      {
       "name": "gain",
       "cells": [
        "-5",
        "1",
        "5",
        "0",
        "-7"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "altitude",
       "cells": [
        "0",
        "-5",
        "-4",
        "1",
        "1"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Leg 5",
     "description": "alt = -6",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "alt": -6,
      "best": 1
     },
     "rows": [
      {
       "name": "gain",
       "cells": [
        "-5",
        "1",
        "5",
        "0",
        "-7"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "altitude",
       "cells": [
        "0",
        "-5",
        "-4",
        "1",
        "1",
        "-6"
       ],
       "hot": [
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Highest point reached.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "best",
       "cells": [
        "1"
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
