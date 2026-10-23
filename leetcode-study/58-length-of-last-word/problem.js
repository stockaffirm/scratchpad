window.PROBLEM = {
 "id": "58",
 "title": "Length of Last Word",
 "level": "Easy",
 "topic": "Array / String",
 "question": "A sentence with spaces. How long is the last word?",
 "why": "'   fly me   to   the moon  ' → last word is 'moon' → 4.",
 "idea": "split() throws away extra spaces; take the last piece.",
 "input": "\"   fly me   to   the moon  \"",
 "output": "4",
 "notes": [],
 "best": "split() throws away extra spaces; take the last piece.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def lengthOfLastWord(self, s):\n        return len(s.split()[-1])",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "split() throws away extra spaces; take the last piece.",
   "frames": [
    {
     "title": "Input",
     "description": "Spaces at the start, between and at the end.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "s",
       "cells": [
        " ",
        " ",
        " ",
        "f",
        "l",
        "y",
        " ",
        "m",
        "e",
        " ",
        " ",
        " ",
        "t",
        "o",
        " ",
        " ",
        " ",
        "t",
        "h",
        "e",
        " ",
        "m",
        "o",
        "o",
        "n",
        " ",
        " "
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "split()",
     "description": "Words only: fly, me, to, the, moon.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "fly",
        "me",
        "to",
        "the",
        "moon"
       ],
       "hot": [
        4
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Last word",
     "description": "[-1] = 'moon'.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "last",
       "cells": [
        "m",
        "o",
        "o",
        "n"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Length",
     "description": "len('moon') = 4",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "4"
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
