window.PROBLEM = {
 "id": "138",
 "title": "Copy List with Random Pointer",
 "level": "Medium",
 "topic": "Linked List",
 "question": "Each node has a 'next' and also a 'random' pointer to any node. Make a deep copy of the whole list.",
 "why": "List A→B→C with random pointers; the copy has new nodes A'→B'→C' with matching pointers.",
 "idea": "Pass 1: make a copy of every node and remember old→new in a dictionary. Pass 2: wire up next and random using the dictionary.",
 "input": "A→B→C, A.random=C, B.random=A, C.random=B",
 "output": "A'→B'→C' with the same random links",
 "notes": [],
 "best": "Pass 1: make a copy of every node and remember old→new in a dictionary. Pass 2: wire up next and random using the dictionary.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def copyRandomList(self, head):\n        clone = {None: None}\n        cur = head\n        while cur:\n            clone[cur] = Node(cur.val); cur = cur.next\n        cur = head\n        while cur:\n            clone[cur].next = clone[cur.next]; clone[cur].random = clone[cur.random]; cur = cur.next\n        return clone[head]",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Pass 1: make a copy of every node and remember old→new in a dictionary. Pass 2: wire up next and random using the dictionary.",
   "frames": [
    {
     "title": "Original",
     "description": "Three nodes with next and random pointers.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "next",
       "cells": [
        "A",
        "B",
        "C"
       ],
       "hot": [],
       "marks": {},
       "kind": "list"
      },
      {
       "name": "random",
       "cells": [
        "A→C",
        "B→A",
        "C→B"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Pass 1",
     "description": "Create A', B', C' with no pointers yet; remember A→A', B→B', C→C'.",
     "codeLines": [
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "clone map",
       "cells": [
        "A→A'",
        "B→B'",
        "C→C'"
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
     "title": "Pass 2",
     "description": "Set A'.next = clone[A.next] = B' and A'.random = clone[A.random] = C'. Same for the others.",
     "codeLines": [
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "copy next",
       "cells": [
        "A'",
        "B'",
        "C'"
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
       "name": "copy random",
       "cells": [
        "A'→C'",
        "B'→A'",
        "C'→B'"
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
     "title": "Return",
     "description": "clone[head] = A'.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "copy",
       "cells": [
        "A'",
        "B'",
        "C'"
       ],
       "hot": [
        0
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
