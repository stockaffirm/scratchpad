window.PROBLEM = {
 "id": "443",
 "title": "String Compression",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Compress runs in place: letter, then the count if it is greater than 1. Return new length.",
 "input": "[\"a\",\"a\",\"b\",\"b\",\"c\",\"c\",\"c\"]",
 "output": "6 → [\"a\",\"2\",\"b\",\"2\",\"c\",\"3\"]",
 "notes": [
  "write never passes read, so overwriting is safe.",
  "Counts above 9 are written digit by digit."
 ],
 "best": "Read/write pointers compress in place with O(1) extra space.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def compress(self, chars):\n        write = read = 0\n        while read < len(chars):\n            ch, count = chars[read], 0\n            while read < len(chars) and chars[read] == ch:\n                read += 1\n                count += 1\n            chars[write] = ch\n            write += 1\n            if count > 1:\n                for d in str(count):\n                    chars[write] = d\n                    write += 1\n        return write",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Read pointer scans a run; write pointer lays down the compressed form.",
   "frames": [
    {
     "title": "Input",
     "description": "Runs: aa, bb, ccc.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [],
       "marks": {
        "0": "read,write"
       }
      }
     ]
    },
    {
     "title": "Run of a",
     "description": "Read 2 × 'a'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "count": 2
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "0": "write",
        "2": "read"
       }
      }
     ]
    },
    {
     "title": "Write a2",
     "description": "Write letter and count at the write pointer.",
     "codeLines": [
      9,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "2": "read,write"
       }
      }
     ]
    },
    {
     "title": "Run of b",
     "description": "Read 2 × 'b'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "count": 2
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {
        "2": "write",
        "4": "read"
       }
      }
     ]
    },
    {
     "title": "Write b2",
     "description": "Write letter and count at the write pointer.",
     "codeLines": [
      9,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "c",
        "c"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {
        "4": "read,write"
       }
      }
     ]
    },
    {
     "title": "Run of c",
     "description": "Read 3 × 'c'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "count": 3
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "c",
        "c"
       ],
       "hot": [
        4,
        5,
        6
       ],
       "marks": {
        "4": "write",
        "7": "read"
       }
      }
     ]
    },
    {
     "title": "Write c3",
     "description": "Write letter and count at the write pointer.",
     "codeLines": [
      9,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "3",
        "c"
       ],
       "hot": [
        4,
        5
       ],
       "marks": {
        "6": "write",
        "7": "read"
       }
      }
     ]
    },
    {
     "title": "Return",
     "description": "Answer is the write position.",
     "codeLines": [
      15
     ],
     "facts": {
      "length": 6
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "3"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
