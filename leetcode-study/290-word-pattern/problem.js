window.PROBLEM = {
 "id": "290",
 "title": "Word Pattern",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Does the text follow the pattern, with each pattern letter standing for exactly one word (and vice versa)?",
 "why": "pattern abba and 'dog cat cat dog': a=dog, b=cat → True.",
 "idea": "Two maps again: letter→word and word→letter.",
 "input": "pattern=\"abba\", s=\"dog cat cat dog\"",
 "output": "True",
 "notes": [],
 "best": "Two maps again: letter→word and word→letter.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def wordPattern(self, pattern, s):\n        w = s.split()\n        if len(w) != len(pattern): return False\n        m1, m2 = {}, {}\n        for p, x in zip(pattern, w):\n            if m1.setdefault(p, x) != x or m2.setdefault(x, p) != p: return False\n        return True",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Two maps again: letter→word and word→letter.",
   "frames": [
    {
     "title": "Start",
     "description": "Split into words.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "pattern",
       "cells": [
        "a",
        "b",
        "b",
        "a"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "words",
       "cells": [
        "dog",
        "cat",
        "cat",
        "dog"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "a ↔ dog",
     "description": "New pair.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "pattern",
       "cells": [
        "a",
        "b",
        "b",
        "a"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "words",
       "cells": [
        "dog",
        "cat",
        "cat",
        "dog"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "map",
       "cells": [
        "a↔dog"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "b ↔ cat",
     "description": "New pair.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "pattern",
       "cells": [
        "a",
        "b",
        "b",
        "a"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "words",
       "cells": [
        "dog",
        "cat",
        "cat",
        "dog"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "map",
       "cells": [
        "a↔dog",
        "b↔cat"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "b ↔ cat, a ↔ dog",
     "description": "Repeats match their earlier pairs.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "pattern",
       "cells": [
        "a",
        "b",
        "b",
        "a"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {}
      },
      {
       "name": "words",
       "cells": [
        "dog",
        "cat",
        "cat",
        "dog"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {}
      },
      {
       "name": "map",
       "cells": [
        "a↔dog",
        "b↔cat"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "True",
     "codeLines": [
      8
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
