window.PROBLEM = {
 "id": "68",
 "title": "Text Justification",
 "level": "Hard",
 "topic": "Array / String",
 "question": "Pack words into lines of exactly maxWidth characters, spreading extra spaces evenly between words. The last line is left-aligned.",
 "why": "Words This, is, an | example, of, text | justification. with width 16.",
 "idea": "Fit as many words as possible per line; then share leftover spaces across the gaps (extras go to the left gaps).",
 "input": "[\"This\",\"is\",\"an\",\"example\",\"of\",\"text\",\"justification.\"], maxWidth=16",
 "output": "[\"This    is    an\",\"example  of text\",\"justification.  \"]",
 "notes": [],
 "best": "Fit as many words as possible per line; then share leftover spaces across the gaps (extras go to the left gaps).",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def fullJustify(self, words, maxWidth):\n        res, line, length = [], [], 0\n        for w in words:\n            if length + len(w) + len(line) > maxWidth:\n                gaps = len(line) - 1 or 1\n                for i in range(maxWidth - length):\n                    line[i % gaps] += ' '\n                res.append(''.join(line)); line, length = [], 0\n            line.append(w); length += len(w)\n        res.append(' '.join(line).ljust(maxWidth))\n        return res",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Fit as many words as possible per line; then share leftover spaces across the gaps (extras go to the left gaps).",
   "frames": [
    {
     "title": "Fill line 1",
     "description": "'This', 'is', 'an' fit (8 letters). Adding 'example' would overflow.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "line",
       "cells": [
        "This",
        "is",
        "an"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "letters",
       "cells": [
        "8"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Spread spaces",
     "description": "16 − 8 = 8 spaces over 2 gaps → 4 each.",
     "codeLines": [
      6,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "This    is    an"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Fill line 2",
     "description": "'example','of','text' = 13 letters + 2 gaps = 15. 'justification.' won't fit.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "line",
       "cells": [
        "example",
        "of",
        "text"
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
     "title": "Spread spaces",
     "description": "3 spaces over 2 gaps → extras go left: 2 then 1.",
     "codeLines": [
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "This    is    an",
        "example  of text"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Last line",
     "description": "Left-align, pad the right.",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "This    is    an",
        "example  of text",
        "justification.  "
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
