window.PROBLEM = {
 "id": "841",
 "title": "Keys and Rooms",
 "level": "Medium",
 "topic": "Graphs - DFS",
 "question": "Room 0 is open. Each room holds keys to other rooms. Can you visit them all?",
 "input": "[[1,3],[3,0,1],[2],[0]]",
 "output": "False",
 "notes": [
  "Rooms are nodes, keys are directed edges.",
  "Unreached room → False."
 ],
 "best": "Plain DFS with a visited set; compare the count at the end.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def canVisitAllRooms(self, rooms):\n        seen, stack = {0}, [0]\n        while stack:\n            r = stack.pop()\n            for k in rooms[r]:\n                if k not in seen:\n                    seen.add(k); stack.append(k)\n        return len(seen) == len(rooms)",
   "time": "O(n+keys)",
   "space": "O(n)",
   "explain": "DFS from room 0 over keys.",
   "frames": [
    {
     "title": "Rooms",
     "description": "Keys per room.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "room: keys",
       "cells": [
        "0:13",
        "1:301",
        "2:2",
        "3:0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Start",
     "description": "Open room 0.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "room: keys",
       "cells": [
        "0:13",
        "1:301",
        "2:2",
        "3:0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "visited",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "0"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Enter room 0",
     "description": "New keys → rooms 1, 3.",
     "codeLines": [
      5,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "room: keys",
       "cells": [
        "0:13",
        "1:301",
        "2:2",
        "3:0"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "3"
       ],
       "hot": [
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "1",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Enter room 3",
     "description": "No new rooms.",
     "codeLines": [
      5,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "room: keys",
       "cells": [
        "0:13",
        "1:301",
        "2:2",
        "3:0"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "stack",
       "cells": [
        "1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Enter room 1",
     "description": "No new rooms.",
     "codeLines": [
      5,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "room: keys",
       "cells": [
        "0:13",
        "1:301",
        "2:2",
        "3:0"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "3"
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
     "title": "Return",
     "description": "Visited 3 of 4 rooms — room 2 is never reachable.",
     "codeLines": [
      9
     ],
     "facts": {
      "result": "False"
     },
     "rows": [
      {
       "name": "visited",
       "cells": [
        "0",
        "1",
        "✗",
        "3"
       ],
       "hot": [
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
