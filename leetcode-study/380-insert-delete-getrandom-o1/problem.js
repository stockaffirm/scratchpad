window.PROBLEM = {
 "id": "380",
 "title": "Insert Delete GetRandom O(1)",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Build a set where insert, remove and getRandom all run in O(1) time.",
 "why": "insert 1, insert 2, remove 1, getRandom (only 2 is left).",
 "idea": "A list for random access + a dictionary of value → index. To remove, swap the last item into the hole.",
 "input": "insert(1), insert(2), remove(1), getRandom()",
 "output": "True, True, True, 2",
 "notes": [],
 "best": "A list for random access + a dictionary of value → index. To remove, swap the last item into the hole.",
 "variants": [
  {
   "name": "Solution",
   "code": "class RandomizedSet:\n    def __init__(self):\n        self.vals, self.pos = [], {}\n    def insert(self, v):\n        if v in self.pos: return False\n        self.pos[v] = len(self.vals); self.vals.append(v); return True\n    def remove(self, v):\n        if v not in self.pos: return False\n        i, last = self.pos[v], self.vals[-1]\n        self.vals[i] = last; self.pos[last] = i\n        self.vals.pop(); del self.pos[v]; return True\n    def getRandom(self):\n        return random.choice(self.vals)",
   "time": "O(1) per op",
   "space": "O(n)",
   "explain": "A list for random access + a dictionary of value → index. To remove, swap the last item into the hole.",
   "frames": [
    {
     "title": "insert(1)",
     "description": "Append 1; remember its index 0.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "vals",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "index of",
       "cells": [
        "1→0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "insert(2)",
     "description": "Append 2; index 1.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "vals",
       "cells": [
        "1",
        "2"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "index of",
       "cells": [
        "1→0",
        "2→1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "remove(1)",
     "description": "1 sits at index 0. Move the last item (2) into slot 0.",
     "codeLines": [
      9,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "vals",
       "cells": [
        "2",
        "2"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      },
      {
       "name": "index of",
       "cells": [
        "2→0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pop",
     "description": "Drop the duplicate tail → list is just [2].",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "vals",
       "cells": [
        "2"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "index of",
       "cells": [
        "2→0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "getRandom()",
     "description": "Pick a random index of the list → 2.",
     "codeLines": [
      13
     ],
     "facts": {},
     "rows": [
      {
       "name": "vals",
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
