window.PROBLEM = {
 "id": "345",
 "title": "Reverse Vowels of a String",
 "level": "Easy",
 "topic": "Two Pointers",
 "question": "Reverse only the vowels (a e i o u, upper or lower case). Every other letter stays where it is.",
 "input": "\"IceCreAm\"",
 "output": "\"AceCreIm\"",
 "notes": [
  "Two pointers scan inward from the left and right ends.",
  "Skip consonants; swap only vowel characters.",
  "Python strings are immutable, so list(s) allows indexed swapping.",
  "Time O(n); the Python list and resulting string require O(n) memory."
 ],
 "best": "Linear scan with two inward-moving pointers.",
 "variants": [
  {
   "name": "Two pointers",
   "code": "class Solution:\n    def reverseVowels(self, s):\n        vowels = set('aeiouAEIOU')\n        a = list(s)\n        left, right = 0, len(a)-1\n        while left < right:\n            while left < right and a[left] not in vowels: left += 1\n            while left < right and a[right] not in vowels: right -= 1\n            a[left], a[right] = a[right], a[left]\n            left, right = left+1, right-1\n        return ''.join(a)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Linear scan with two inward-moving pointers.",
   "frames": [
    {
     "title": "Input",
     "description": "Vowels in IceCreAm: I, e, e, A.",
     "codeLines": [
      3,
      4,
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "characters",
       "cells": [
        "I",
        "c",
        "e",
        "C",
        "r",
        "e",
        "A",
        "m"
       ],
       "hot": [
        0,
        2,
        5,
        6
       ]
      }
     ]
    },
    {
     "title": "Move right pointer",
     "description": "right=7 is m, not a vowel. Move to right=6 (A).",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "characters",
       "cells": [
        "I",
        "c",
        "e",
        "C",
        "r",
        "e",
        "A",
        "m"
       ],
       "hot": [
        0,
        6
       ]
      }
     ]
    },
    {
     "title": "Swap I and A",
     "description": "Swap at indexes 0 and 6; move pointers inward.",
     "codeLines": [
      9,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "characters",
       "cells": [
        "A",
        "c",
        "e",
        "C",
        "r",
        "e",
        "I",
        "m"
       ],
       "hot": [
        0,
        6
       ]
      }
     ]
    },
    {
     "title": "Find inner vowels",
     "description": "Skip consonant c; find e at index 2 and e at index 5.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "characters",
       "cells": [
        "A",
        "c",
        "e",
        "C",
        "r",
        "e",
        "I",
        "m"
       ],
       "hot": [
        2,
        5
       ]
      }
     ],
     "sub": [
      {
       "description": "left = 1: 'c' is not a vowel → skip, left = 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "characters",
         "cells": [
          "A",
          "c",
          "e",
          "C",
          "r",
          "e",
          "I",
          "m"
         ],
         "hot": [
          1
         ],
         "marks": {
          "1": "left",
          "5": "right"
         }
        }
       ]
      },
      {
       "description": "left = 2: 'e' is a vowel → stop",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "characters",
         "cells": [
          "A",
          "c",
          "e",
          "C",
          "r",
          "e",
          "I",
          "m"
         ],
         "hot": [
          2
         ],
         "marks": {
          "2": "left",
          "5": "right"
         }
        }
       ]
      },
      {
       "description": "right = 5: 'e' is a vowel → stop",
       "codeLines": [
        8
       ],
       "facts": {},
       "rows": [
        {
         "name": "characters",
         "cells": [
          "A",
          "c",
          "e",
          "C",
          "r",
          "e",
          "I",
          "m"
         ],
         "hot": [
          2,
          5
         ],
         "marks": {
          "2": "left",
          "5": "right"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Finish",
     "description": "Swapping e and e makes no change. Return modified string.",
     "codeLines": [
      9,
      10,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "A",
        "c",
        "e",
        "C",
        "r",
        "e",
        "I",
        "m"
       ],
       "hot": [
        0,
        2,
        5,
        6
       ]
      }
     ]
    }
   ]
  }
 ],
 "why": "IceCreAm: the vowels in order are I, e, e, A. Reversed they are A, e, e, I. Put them back → AceCreIm.",
 "idea": "Two pointers from both ends. Skip non-vowels. Swap vowels."
};
