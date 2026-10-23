window.PROBLEM = {
 "id": "383",
 "title": "Ransom Note",
 "level": "Easy",
 "topic": "Hashmap",
 "question": "Can you build the ransom note using letters cut from the magazine (each letter used once)?",
 "why": "note 'aa', magazine 'aab' → the magazine has two a's → True.",
 "idea": "Count letters in both; the note's counts must fit inside the magazine's.",
 "input": "ransomNote=\"aa\", magazine=\"aab\"",
 "output": "True",
 "notes": [],
 "best": "Count letters in both; the note's counts must fit inside the magazine's.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def canConstruct(self, ransomNote, magazine):\n        return not (Counter(ransomNote) - Counter(magazine))",
   "time": "O(n+m)",
   "space": "O(1)",
   "explain": "Count letters in both; the note's counts must fit inside the magazine's.",
   "frames": [
    {
     "title": "Count note",
     "description": "a:2",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "note",
       "cells": [
        "a:2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Count magazine",
     "description": "a:2, b:1",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "magazine",
       "cells": [
        "a:2",
        "b:1"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Subtract",
     "description": "note − magazine = nothing left over.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "left over",
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
     "description": "True",
     "codeLines": [
      3
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
