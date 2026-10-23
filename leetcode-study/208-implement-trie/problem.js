window.PROBLEM = {
 "id": "208",
 "title": "Implement Trie (Prefix Tree)",
 "level": "Medium",
 "topic": "Trie",
 "question": "Build a trie with insert, search (whole word) and startsWith (prefix).",
 "input": "insert(apple), search(apple), search(app), startsWith(app), insert(app), search(app)",
 "output": "T, F, T, T",
 "notes": [
  "search needs the '$' end marker; startsWith doesn't.",
  "Shared prefixes share nodes."
 ],
 "best": "Dict-of-dicts trie; '$' separates 'is a prefix' from 'is a word'.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Trie:\n    def __init__(self): self.root = {}\n    def insert(self, word):\n        node = self.root\n        for ch in word: node = node.setdefault(ch, {})\n        node['$'] = True\n    def search(self, word):\n        node = self._walk(word)\n        return bool(node and '$' in node)\n    def startsWith(self, prefix):\n        return self._walk(prefix) is not None\n    def _walk(self, s):\n        node = self.root\n        for ch in s:\n            if ch not in node: return None\n            node = node[ch]\n        return node",
   "time": "O(len) per op",
   "space": "O(total letters)",
   "explain": "Nested dicts, one level per letter; '$' marks a word end.",
   "frames": [
    {
     "title": "insert('apple')",
     "description": "One node per letter, then an end marker.",
     "codeLines": [
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "trie",
       "cells": [
        "a",
        "p",
        "p",
        "l",
        "e",
        "$"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "search('apple')",
     "description": "Path exists and ends with $ → True.",
     "codeLines": [
      9
     ],
     "facts": {
      "result": "True"
     },
     "rows": [
      {
       "name": "walk",
       "cells": [
        "a",
        "p",
        "p",
        "l",
        "e",
        "$"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "search('app')",
     "description": "Path exists, but no $ after the 2nd p → False.",
     "codeLines": [
      9
     ],
     "facts": {
      "result": "False"
     },
     "rows": [
      {
       "name": "walk",
       "cells": [
        "a",
        "p",
        "p"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "after p",
       "cells": [
        "l",
        "(no $)"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "startsWith('app')",
     "description": "Path exists → True.",
     "codeLines": [
      11
     ],
     "facts": {
      "result": "True"
     },
     "rows": [
      {
       "name": "walk",
       "cells": [
        "a",
        "p",
        "p"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "insert('app')",
     "description": "Reuse a-p-p, add $ after the 2nd p.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "trie",
       "cells": [
        "a",
        "p",
        "p",
        "$"
       ],
       "hot": [
        3
       ],
       "marks": {},
       "kind": "list"
      }
     ]
    },
    {
     "title": "search('app')",
     "description": "Now $ exists → True.",
     "codeLines": [
      9
     ],
     "facts": {
      "result": "True"
     },
     "rows": [
      {
       "name": "walk",
       "cells": [
        "a",
        "p",
        "p",
        "$"
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
    }
   ]
  }
 ]
};
