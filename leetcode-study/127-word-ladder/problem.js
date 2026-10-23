window.PROBLEM = {
 "id": "127",
 "title": "Word Ladder",
 "level": "Hard",
 "topic": "Graph - BFS",
 "question": "Change one letter at a time to turn beginWord into endWord. Every step must be a word in the list. How many words are in the shortest sequence?",
 "why": "hit → hot → dot → dog → cog is 5 words.",
 "idea": "BFS: from each word, try replacing each letter with a–z and keep words from the list.",
 "input": "begin=\"hit\", end=\"cog\", words=[hot,dot,dog,lot,log,cog]",
 "output": "5",
 "notes": [],
 "best": "BFS: from each word, try replacing each letter with a–z and keep words from the list.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def ladderLength(self, begin, end, words):\n        words, q = set(words), deque([(begin, 1)])\n        while q:\n            w, d = q.popleft()\n            if w == end: return d\n            for i in range(len(w)):\n                for ch in 'abcdefghijklmnopqrstuvwxyz':\n                    nxt = w[:i] + ch + w[i+1:]\n                    if nxt in words:\n                        words.remove(nxt); q.append((nxt, d + 1))\n        return 0",
   "time": "O(n·L·26)",
   "space": "O(n)",
   "explain": "BFS: from each word, try replacing each letter with a–z and keep words from the list.",
   "frames": [
    {
     "title": "Start",
     "description": "hit (length 1).",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 1",
       "cells": [
        "hit"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 2",
     "description": "hit → hot",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 1",
       "cells": [
        "hit"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "level 2",
       "cells": [
        "hot"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Level 3",
     "description": "hot → dot, lot",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 3",
       "cells": [
        "dot",
        "lot"
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
     "title": "Level 4",
     "description": "dot → dog; lot → log",
     "codeLines": [
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 4",
       "cells": [
        "dog",
        "log"
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
     "title": "Level 5",
     "description": "dog → cog = end.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "level 5",
       "cells": [
        "cog"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "5 words in the shortest ladder.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "path",
       "cells": [
        "hit",
        "hot",
        "dot",
        "dog",
        "cog"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    }
   ]
  }
 ]
};
