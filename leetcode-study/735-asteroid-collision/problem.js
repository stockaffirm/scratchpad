window.PROBLEM = {
 "id": "735",
 "title": "Asteroid Collision",
 "level": "Medium",
 "topic": "Stack",
 "question": "Positive moves right, negative moves left. Smaller explodes; equal both explode. Final state?",
 "input": "[5, 10, -5, -12]",
 "output": "[-12]",
 "notes": [
  "Collision needs stack top > 0 and new a < 0.",
  "while/else: else runs only if no break."
 ],
 "best": "Stack of survivors; collisions only happen at the top.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def asteroidCollision(self, asteroids):\n        stack = []\n        for a in asteroids:\n            while stack and a < 0 < stack[-1]:\n                if stack[-1] < -a:\n                    stack.pop(); continue\n                if stack[-1] == -a: stack.pop()\n                break\n            else:\n                stack.append(a)\n        return stack",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Only a left-mover meeting a right-mover on top collides.",
   "frames": [
    {
     "title": "Input",
     "description": "5 and 10 go right; -5 and -12 go left.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "asteroids",
       "cells": [
        "5",
        "10",
        "-5",
        "-12"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "a = 5",
     "description": "No collision. Push 5.",
     "codeLines": [
      5,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "asteroids",
       "cells": [
        "5",
        "10",
        "-5",
        "-12"
       ],
       "hot": [
        0
       ],
       "marks": {
        "0": "a"
       }
      },
      {
       "name": "stack",
       "cells": [
        "5"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "a = 10",
     "description": "No collision. Push 10.",
     "codeLines": [
      5,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "asteroids",
       "cells": [
        "5",
        "10",
        "-5",
        "-12"
       ],
       "hot": [
        1
       ],
       "marks": {
        "1": "a"
       }
      },
      {
       "name": "stack",
       "cells": [
        "5",
        "10"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "a = -5",
     "description": "New -5 destroyed. ",
     "codeLines": [
      5,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "asteroids",
       "cells": [
        "5",
        "10",
        "-5",
        "-12"
       ],
       "hot": [
        2
       ],
       "marks": {
        "2": "a"
       }
      },
      {
       "name": "stack",
       "cells": [
        "5",
        "10"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "a = -12",
     "description": "Top 10 destroyed. Top 5 destroyed. Push -12.",
     "codeLines": [
      5,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "asteroids",
       "cells": [
        "5",
        "10",
        "-5",
        "-12"
       ],
       "hot": [
        3
       ],
       "marks": {
        "3": "a"
       }
      },
      {
       "name": "stack",
       "cells": [
        "-12"
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
     "description": "Survivors.",
     "codeLines": [
      12
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "-12"
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
