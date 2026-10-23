window.PROBLEM = {
 "id": "135",
 "title": "Candy",
 "level": "Hard",
 "topic": "Array / String",
 "question": "Children in a line with ratings. Everyone gets at least 1 candy, and a child with a higher rating than a neighbour gets more candy than that neighbour. Fewest candies?",
 "why": "ratings [1,0,2] → candies [2,1,2] = 5.",
 "idea": "Two passes: left→right fixes 'higher than left neighbour', right→left fixes 'higher than right neighbour'.",
 "input": "ratings=[1,0,2]",
 "output": "5",
 "notes": [],
 "best": "Two passes: left→right fixes 'higher than left neighbour', right→left fixes 'higher than right neighbour'.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def candy(self, ratings):\n        n = len(ratings); c = [1] * n\n        for i in range(1, n):\n            if ratings[i] > ratings[i-1]: c[i] = c[i-1] + 1\n        for i in range(n - 2, -1, -1):\n            if ratings[i] > ratings[i+1]: c[i] = max(c[i], c[i+1] + 1)\n        return sum(c)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Two passes: left→right fixes 'higher than left neighbour', right→left fixes 'higher than right neighbour'.",
   "frames": [
    {
     "title": "Start",
     "description": "Everyone gets 1.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "ratings",
       "cells": [
        "1",
        "0",
        "2"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "candy",
       "cells": [
        "1",
        "1",
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Left → right",
     "description": "Child 2 (rating 2) beats the left neighbour → 2 candies.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "ratings",
       "cells": [
        "1",
        "0",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "candy",
       "cells": [
        "1",
        "1",
        "2"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Right → left",
     "description": "Child 0 (rating 1) beats the right neighbour (0) → needs more than 1 → 2.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "ratings",
       "cells": [
        "1",
        "0",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "candy",
       "cells": [
        "2",
        "1",
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "2 + 1 + 2 = 5.",
     "codeLines": [
      8
     ],
     "facts": {
      "total": 5
     },
     "rows": [
      {
       "name": "candy",
       "cells": [
        "2",
        "1",
        "2"
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
