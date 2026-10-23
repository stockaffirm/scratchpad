window.PROBLEM = {
 "id": "1207",
 "title": "Unique Number of Occurrences",
 "level": "Easy",
 "topic": "Hash Map / Set",
 "question": "Is every value's occurrence count different?",
 "input": "[1,2,2,1,1,3]",
 "output": "True",
 "notes": [
  "If counts are unique, the set keeps the same size.",
  "Counter is a dict of value → frequency."
 ],
 "best": "Counter + set of counts; compare sizes.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def uniqueOccurrences(self, arr):\n        count = Counter(arr)\n        return len(set(count.values())) == len(count)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Count, then check the counts are all distinct.",
   "frames": [
    {
     "title": "Input",
     "description": "Count each value.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "arr",
       "cells": [
        "1",
        "2",
        "2",
        "1",
        "1",
        "3"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Counter",
     "description": "1→3, 2→2, 3→1",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "value",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "count",
       "cells": [
        "3",
        "2",
        "1"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Set of counts",
     "description": "{3,2,1} has 3 items, same as the 3 values.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "counts",
       "cells": [
        "3",
        "2",
        "1"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "set",
       "cells": [
        "3",
        "2",
        "1"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "All counts are different.",
     "codeLines": [
      4
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
