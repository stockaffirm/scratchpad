window.PROBLEM = {
 "id": "605",
 "title": "Can Place Flowers",
 "level": "Easy",
 "topic": "Greedy",
 "question": "A row of plots: 1 means a flower, 0 means empty. You can't plant in a plot next to a flower. Can you plant n more flowers?",
 "input": "flowerbed=[1,0,0,0,1], n=1",
 "output": "True",
 "notes": [
  "Greedy means take the earliest locally valid planting decision, without backtracking.",
  "OR short circuits: i==0 prevents accidentally looking up flowerbed[-1].",
  "AND ensures both neighbors are empty (or outside the boundaries).",
  "Early return improves practical speed but worst-case time remains O(m).",
  "Check n == 0 first to avoid unnecessary mutation of flowerbed."
 ],
 "best": "Choose the earliest legal position and stop once enough flowers are planted.",
 "variants": [
  {
   "name": "Greedy + early return",
   "code": "class Solution:\n    def canPlaceFlowers(self, flowerbed, n):\n        if n == 0: return True\n        count = 0\n        for i in range(len(flowerbed)):\n            if flowerbed[i] == 0:\n                left = i == 0 or flowerbed[i-1] == 0\n                right = i == len(flowerbed)-1 or flowerbed[i+1] == 0\n                if left and right:\n                    flowerbed[i] = 1\n                    count += 1\n                    if count >= n: return True\n        return False",
   "time": "O(m)",
   "space": "O(1) auxiliary",
   "explain": "Choose the earliest legal position and stop once enough flowers are planted.",
   "frames": [
    {
     "title": "Input",
     "description": "Need to plant one flower without adjacent flowers.",
     "codeLines": [
      2,
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "plots",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "i=0: skip",
     "description": "The first plot is already occupied.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "plots",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": [
        0
       ]
      }
     ]
    },
    {
     "title": "i=1: blocked",
     "description": "Index 1 is empty but left neighbor index 0 contains 1.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "plots",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": [
        0,
        1
       ]
      }
     ]
    },
    {
     "title": "i=2: can plant",
     "description": "Index 2 and both neighbors are 0.",
     "codeLines": [
      7,
      8,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "plots",
       "cells": [
        "1",
        "0",
        "0",
        "0",
        "1"
       ],
       "hot": [
        1,
        2,
        3
       ]
      }
     ]
    },
    {
     "title": "Plant and return",
     "description": "Change index 2 to 1. count = 1, so return True.",
     "codeLines": [
      10,
      11,
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "plots",
       "cells": [
        "1",
        "0",
        "1",
        "0",
        "1"
       ],
       "hot": [
        2
       ]
      }
     ]
    }
   ]
  }
 ],
 "why": "[1,0,0,0,1]: only the middle plot has empty plots on both sides. Plant there → one flower fits → True.",
 "idea": "Go left to right. Plant whenever a plot and both its neighbours are empty."
};
