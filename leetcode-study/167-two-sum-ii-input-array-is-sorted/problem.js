window.PROBLEM = {
 "id": "167",
 "title": "Two Sum II - Input Array Is Sorted",
 "level": "Medium",
 "topic": "Two Pointers",
 "question": "The list is sorted. Find two numbers that add up to the target and return their positions (counting from 1).",
 "why": "[2,7,11,15], target 9: 2 + 7 = 9 → positions 1 and 2.",
 "idea": "Pointers at both ends: sum too small → move left up; too big → move right down.",
 "input": "[2,7,11,15], target=9",
 "output": "[1, 2]",
 "notes": [],
 "best": "Pointers at both ends: sum too small → move left up; too big → move right down.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def twoSum(self, numbers, target):\n        l, r = 0, len(numbers) - 1\n        while l < r:\n            s = numbers[l] + numbers[r]\n            if s == target: return [l + 1, r + 1]\n            if s < target: l += 1\n            else: r -= 1",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Pointers at both ends: sum too small → move left up; too big → move right down.",
   "frames": [
    {
     "title": "Start",
     "description": "Ends of the sorted list.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "numbers",
       "cells": [
        "2",
        "7",
        "11",
        "15"
       ],
       "hot": [],
       "marks": {
        "0": "l",
        "3": "r"
       }
      }
     ]
    },
    {
     "title": "2 + 15 = 17",
     "description": "Too big → move r left.",
     "codeLines": [
      5,
      8
     ],
     "facts": {
      "sum": 17
     },
     "rows": [
      {
       "name": "numbers",
       "cells": [
        "2",
        "7",
        "11",
        "15"
       ],
       "hot": [
        0,
        3
       ],
       "marks": {
        "0": "l",
        "3": "r"
       }
      }
     ]
    },
    {
     "title": "2 + 11 = 13",
     "description": "Too big → move r left.",
     "codeLines": [
      5,
      8
     ],
     "facts": {
      "sum": 13
     },
     "rows": [
      {
       "name": "numbers",
       "cells": [
        "2",
        "7",
        "11",
        "15"
       ],
       "hot": [
        0,
        2
       ],
       "marks": {
        "0": "l",
        "2": "r"
       }
      }
     ]
    },
    {
     "title": "2 + 7 = 9",
     "description": "Equals target → positions 1 and 2.",
     "codeLines": [
      6
     ],
     "facts": {
      "sum": 9
     },
     "rows": [
      {
       "name": "numbers",
       "cells": [
        "2",
        "7",
        "11",
        "15"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "0": "l",
        "1": "r"
       }
      }
     ]
    }
   ]
  }
 ]
};
