window.PROBLEM = {
 "id": "128",
 "title": "Longest Consecutive Sequence",
 "level": "Medium",
 "topic": "Hashmap",
 "question": "Find the longest run of consecutive integers (like 1,2,3,4) in the unsorted list, in O(n).",
 "why": "[100,4,200,1,3,2] contains 1,2,3,4 → length 4.",
 "idea": "Put numbers in a set. Only start counting from numbers with no smaller neighbour.",
 "input": "[100,4,200,1,3,2]",
 "output": "4",
 "notes": [],
 "best": "Put numbers in a set. Only start counting from numbers with no smaller neighbour.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def longestConsecutive(self, nums):\n        s, best = set(nums), 0\n        for x in s:\n            if x - 1 not in s:\n                y = x\n                while y + 1 in s: y += 1\n                best = max(best, y - x + 1)\n        return best",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Put numbers in a set. Only start counting from numbers with no smaller neighbour.",
   "frames": [
    {
     "title": "Set",
     "description": "Fast lookups.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "set",
       "cells": [
        "100",
        "4",
        "200",
        "1",
        "3",
        "2"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Skip 4, 3, 2",
     "description": "They have a smaller neighbour — not the start of a run.",
     "codeLines": [
      5
     ],
     "facts": {},
     "rows": [
      {
       "name": "set",
       "cells": [
        "100",
        "4",
        "200",
        "1",
        "3",
        "2"
       ],
       "hot": [
        1,
        4,
        5
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Start at 1",
     "description": "1 has no 0 → run 1,2,3,4 → length 4.",
     "codeLines": [
      7
     ],
     "facts": {
      "best": 4
     },
     "rows": [
      {
       "name": "run",
       "cells": [
        "1",
        "2",
        "3",
        "4"
       ],
       "hot": [
        0,
        1,
        2,
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "Starts 100, 200",
     "description": "Runs of length 1.",
     "codeLines": [
      8
     ],
     "facts": {
      "best": 4
     },
     "rows": [
      {
       "name": "set",
       "cells": [
        "100",
        "4",
        "200",
        "1",
        "3",
        "2"
       ],
       "hot": [
        0,
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "4",
     "codeLines": [
      9
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
