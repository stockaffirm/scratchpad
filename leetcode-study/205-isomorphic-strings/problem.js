window.PROBLEM = {
 "id": "205",
 "title": "Isomorphic Strings",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Can you swap letters in s to get t, with each letter always mapping to the same letter (and no two letters sharing a target)?",
 "why": "egg → add: e→a, g→d works → True.",
 "idea": "Keep two maps (s→t and t→s); any conflict means False.",
 "input": "s=\"egg\", t=\"add\"",
 "output": "True",
 "notes": [],
 "best": "Keep two maps (s→t and t→s); any conflict means False.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def isIsomorphic(self, s, t):\n        m1, m2 = {}, {}\n        for a, b in zip(s, t):\n            if m1.setdefault(a, b) != b or m2.setdefault(b, a) != a: return False\n        return True",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Keep two maps (s→t and t→s); any conflict means False.",
   "frames": [
    {
     "title": "Start",
     "description": "Compare letter by letter.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "e",
        "g",
        "g"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "t",
       "cells": [
        "a",
        "d",
        "d"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "e → a",
     "description": "New pair.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "e",
        "g",
        "g"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "t",
       "cells": [
        "a",
        "d",
        "d"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "map",
       "cells": [
        "e→a"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "g → d",
     "description": "New pair.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "e",
        "g",
        "g"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "t",
       "cells": [
        "a",
        "d",
        "d"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "map",
       "cells": [
        "e→a",
        "g→d"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "g → d again",
     "description": "Same as before → consistent.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "e",
        "g",
        "g"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "t",
       "cells": [
        "a",
        "d",
        "d"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "map",
       "cells": [
        "e→a",
        "g→d"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "No conflicts → True.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "True"
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
