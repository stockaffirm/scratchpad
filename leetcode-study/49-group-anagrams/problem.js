window.PROBLEM = {
 "id": "49",
 "title": "Group Anagrams",
 "level": "Medium",
 "topic": "Hashmap",
 "question": "Group together the words that are anagrams of each other.",
 "why": "eat, tea, ate share letters a-e-t; tan, nat share a-n-t; bat is alone.",
 "idea": "Sort each word's letters to get a key; words with the same key go in the same group.",
 "input": "[\"eat\",\"tea\",\"tan\",\"ate\",\"nat\",\"bat\"]",
 "output": "[[\"eat\",\"tea\",\"ate\"],[\"tan\",\"nat\"],[\"bat\"]]",
 "notes": [],
 "best": "Sort each word's letters to get a key; words with the same key go in the same group.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def groupAnagrams(self, strs):\n        groups = defaultdict(list)\n        for w in strs:\n            groups[''.join(sorted(w))].append(w)\n        return list(groups.values())",
   "time": "O(n·k log k)",
   "space": "O(n·k)",
   "explain": "Sort each word's letters to get a key; words with the same key go in the same group.",
   "frames": [
    {
     "title": "Start",
     "description": "Key = sorted letters.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'eat' → key 'aet'",
     "description": "Add to group 'aet'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "'aet'",
       "cells": [
        "eat"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'tea' → key 'aet'",
     "description": "Add to group 'aet'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "'aet'",
       "cells": [
        "eat",
        "tea"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'tan' → key 'ant'",
     "description": "Add to group 'ant'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "'aet'",
       "cells": [
        "eat",
        "tea"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "'ant'",
       "cells": [
        "tan"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'ate' → key 'aet'",
     "description": "Add to group 'aet'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "'aet'",
       "cells": [
        "eat",
        "tea",
        "ate"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "'ant'",
       "cells": [
        "tan"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'nat' → key 'ant'",
     "description": "Add to group 'ant'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "'aet'",
       "cells": [
        "eat",
        "tea",
        "ate"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "'ant'",
       "cells": [
        "tan",
        "nat"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'bat' → key 'abt'",
     "description": "Add to group 'abt'.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "eat",
        "tea",
        "tan",
        "ate",
        "nat",
        "bat"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "'aet'",
       "cells": [
        "eat",
        "tea",
        "ate"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "'ant'",
       "cells": [
        "tan",
        "nat"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "'abt'",
       "cells": [
        "bat"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "3 groups.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "[eat,tea,ate]",
        "[tan,nat]",
        "[bat]"
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
