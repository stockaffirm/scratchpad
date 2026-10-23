window.PROBLEM = {
 "id": "30",
 "title": "Substring with Concatenation of All Words",
 "level": "Hard",
 "topic": "Sliding Window",
 "question": "Find every start position where the text contains all the given words (same length), joined in any order with nothing in between.",
 "why": "s = 'barfoothefoobarman', words foo + bar: 'barfoo' starts at 0, 'foobar' starts at 9 → [0, 9].",
 "idea": "Slice the text into word-sized chunks from each start and compare their counts with the word counts.",
 "input": "s=\"barfoothefoobarman\", words=[\"foo\",\"bar\"]",
 "output": "[0, 9]",
 "notes": [],
 "best": "Slice the text into word-sized chunks from each start and compare their counts with the word counts.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def findSubstring(self, s, words):\n        w, n = len(words[0]), len(words); need = Counter(words); res = []\n        for i in range(len(s) - w * n + 1):\n            seen = Counter(s[i + j*w : i + (j+1)*w] for j in range(n))\n            if seen == need: res.append(i)\n        return res",
   "time": "O(len·n·w)",
   "space": "O(n)",
   "explain": "Slice the text into word-sized chunks from each start and compare their counts with the word counts.",
   "frames": [
    {
     "title": "Setup",
     "description": "Words have length 3; the target window is 6 characters.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "r",
        "f",
        "o",
        "o",
        "t",
        "h",
        "e",
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        "m",
        "a",
        "n"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "need",
       "cells": [
        "foo:1",
        "bar:1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 0",
     "description": "chunks: 'bar','foo' → same counts → save 0.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "r",
        "f",
        "o",
        "o",
        "t",
        "h",
        "e",
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        "m",
        "a",
        "n"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {
        "0": "i"
       }
      },
      {
       "name": "chunks",
       "cells": [
        "bar",
        "foo"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 1",
     "description": "chunks: 'arf','oot' → no match.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "r",
        "f",
        "o",
        "o",
        "t",
        "h",
        "e",
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        "m",
        "a",
        "n"
       ],
       "hot": [
        1,
        2,
        3,
        4,
        5,
        6
       ],
       "marks": {
        "1": "i"
       }
      },
      {
       "name": "chunks",
       "cells": [
        "arf",
        "oot"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "i = 9",
     "description": "chunks: 'foo','bar' → same counts → save 9.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        "b",
        "a",
        "r",
        "f",
        "o",
        "o",
        "t",
        "h",
        "e",
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        "m",
        "a",
        "n"
       ],
       "hot": [
        9,
        10,
        11,
        12,
        13,
        14
       ],
       "marks": {
        "9": "i"
       }
      },
      {
       "name": "chunks",
       "cells": [
        "foo",
        "bar"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "[0, 9]",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "0",
        "9"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
