window.PROBLEM = {
 "id": "933",
 "title": "Number of Recent Calls",
 "level": "Easy",
 "topic": "Queue",
 "question": "ping(t) returns how many pings happened in the last 3000 ms (inclusive).",
 "input": "ping(1), ping(100), ping(3001), ping(3002)",
 "output": "1, 2, 3, 3",
 "notes": [
  "Each timestamp is pushed and popped once.",
  "Window is [t-3000, t], inclusive."
 ],
 "best": "A queue: times arrive in order, so old ones expire from the front.",
 "variants": [
  {
   "name": "Solution",
   "code": "class RecentCounter:\n    def __init__(self):\n        self.q = deque()\n    def ping(self, t):\n        self.q.append(t)\n        while self.q[0] < t - 3000:\n            self.q.popleft()\n        return len(self.q)",
   "time": "O(1) amortised",
   "space": "O(window)",
   "explain": "Queue of timestamps; drop the expired ones from the front.",
   "frames": [
    {
     "title": "Empty queue",
     "description": "Timestamps arrive in increasing order.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "queue",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "ping(1)",
     "description": "Nothing expired (cut-off -2999).",
     "codeLines": [
      5,
      6,
      8
     ],
     "facts": {
      "cut-off": -2999,
      "count": 1
     },
     "rows": [
      {
       "name": "queue",
       "cells": [
        "1"
       ],
       "hot": [
        0
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "ping(100)",
     "description": "Nothing expired (cut-off -2900).",
     "codeLines": [
      5,
      6,
      8
     ],
     "facts": {
      "cut-off": -2900,
      "count": 2
     },
     "rows": [
      {
       "name": "queue",
       "cells": [
        "1",
        "100"
       ],
       "hot": [
        1
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "ping(3001)",
     "description": "Nothing expired (cut-off 1).",
     "codeLines": [
      5,
      6,
      8
     ],
     "facts": {
      "cut-off": 1,
      "count": 3
     },
     "rows": [
      {
       "name": "queue",
       "cells": [
        "1",
        "100",
        "3001"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "ping(3002)",
     "description": "Window starts at 2: drop 1.",
     "codeLines": [
      5,
      6,
      8
     ],
     "facts": {
      "cut-off": 2,
      "count": 3
     },
     "rows": [
      {
       "name": "queue",
       "cells": [
        "100",
        "3001",
        "3002"
       ],
       "hot": [
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    }
   ]
  }
 ]
};
