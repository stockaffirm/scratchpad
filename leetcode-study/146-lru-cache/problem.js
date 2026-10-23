window.PROBLEM = {
 "id": "146",
 "title": "LRU Cache",
 "level": "Medium",
 "topic": "Linked List",
 "question": "A cache with a fixed size. When it's full, throw out the Least Recently Used item. get and put both count as 'using' an item.",
 "why": "Capacity 2: put 1, put 2, get 1, put 3 → evicts 2 (1 was used more recently), get 2 → −1.",
 "idea": "Keep items in usage order (an ordered dictionary): move to the end when used, drop from the front when full.",
 "input": "cap=2: put(1), put(2), get(1), put(3), get(2)",
 "output": "1, -1",
 "notes": [],
 "best": "Keep items in usage order (an ordered dictionary): move to the end when used, drop from the front when full.",
 "variants": [
  {
   "name": "Solution",
   "code": "class LRUCache:\n    def __init__(self, capacity):\n        self.cap, self.d = capacity, OrderedDict()\n    def get(self, k):\n        if k not in self.d: return -1\n        self.d.move_to_end(k); return self.d[k]\n    def put(self, k, v):\n        self.d[k] = v; self.d.move_to_end(k)\n        if len(self.d) > self.cap: self.d.popitem(last=False)",
   "time": "O(1) per op",
   "space": "O(capacity)",
   "explain": "Keep items in usage order (an ordered dictionary): move to the end when used, drop from the front when full.",
   "frames": [
    {
     "title": "put(1), put(2)",
     "description": "Oldest on the left, newest on the right.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "order (old → new)",
       "cells": [
        "1",
        "2"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "get(1)",
     "description": "1 is used → moves to the newest end.",
     "codeLines": [
      6
     ],
     "facts": {
      "returns": 1
     },
     "rows": [
      {
       "name": "order (old → new)",
       "cells": [
        "2",
        "1"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "put(3)",
     "description": "Cache would hold 3 items → evict the oldest (2).",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "order (old → new)",
       "cells": [
        "1",
        "3"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "get(2)",
     "description": "2 was evicted → −1.",
     "codeLines": [
      5
     ],
     "facts": {
      "returns": -1
     },
     "rows": [
      {
       "name": "order (old → new)",
       "cells": [
        "1",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
