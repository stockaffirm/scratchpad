window.PROBLEM = {
 "id": "134",
 "title": "Gas Station",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Gas stations in a circle. gas[i] is fuel you get at station i, cost[i] is fuel to drive to the next one. Which station can you start from to complete a lap? (−1 if none)",
 "why": "gas [1,2,3,4,5], cost [3,4,5,1,2] → start at station 3.",
 "idea": "If the tank goes negative, the start can't be anywhere before here — restart from the next station.",
 "input": "gas=[1,2,3,4,5], cost=[3,4,5,1,2]",
 "output": "3",
 "notes": [],
 "best": "If the tank goes negative, the start can't be anywhere before here — restart from the next station.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def canCompleteCircuit(self, gas, cost):\n        if sum(gas) < sum(cost): return -1\n        tank = start = 0\n        for i in range(len(gas)):\n            tank += gas[i] - cost[i]\n            if tank < 0:\n                start, tank = i + 1, 0\n        return start",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "If the tank goes negative, the start can't be anywhere before here — restart from the next station.",
   "frames": [
    {
     "title": "Net fuel",
     "description": "gas − cost at each station.",
     "codeLines": [
      3
     ],
     "facts": {
      "total": "≥ 0 → possible"
     },
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Station 0",
     "description": "Tank negative → restart from station 1",
     "codeLines": [
      6,
      8
     ],
     "facts": {
      "tank": 0
     },
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [
        0
       ],
       "marks": {
        "1": "start"
       }
      }
     ]
    },
    {
     "title": "Station 1",
     "description": "Tank negative → restart from station 2",
     "codeLines": [
      6,
      8
     ],
     "facts": {
      "tank": 0
     },
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {
        "2": "start"
       }
      }
     ]
    },
    {
     "title": "Station 2",
     "description": "Tank negative → restart from station 3",
     "codeLines": [
      6,
      8
     ],
     "facts": {
      "tank": 0
     },
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [
        2
       ],
       "marks": {
        "3": "start"
       }
      }
     ]
    },
    {
     "title": "Station 3",
     "description": "tank = 3",
     "codeLines": [
      6
     ],
     "facts": {
      "tank": 3
     },
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "start"
       }
      }
     ]
    },
    {
     "title": "Station 4",
     "description": "tank = 6",
     "codeLines": [
      6
     ],
     "facts": {
      "tank": 6
     },
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [
        4
       ],
       "marks": {
        "3": "start"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Start at station 3.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "gas − cost",
       "cells": [
        "-2",
        "-2",
        "-2",
        "3",
        "3"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "start"
       }
      }
     ]
    }
   ]
  }
 ]
};
