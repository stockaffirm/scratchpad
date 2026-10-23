window.PROBLEM = {
 "id": "1268",
 "title": "Search Suggestions System",
 "level": "Medium",
 "topic": "Trie",
 "question": "After each typed letter, suggest up to 3 products with that prefix, in lexicographic order.",
 "input": "products=[mobile, mouse, moneypot, monitor, mousepad], word=\"mouse\"",
 "output": "[[mobile,moneypot,monitor], same, [mouse,mousepad], same, same]",
 "notes": [
  "Sorted order = lexicographic suggestion order.",
  "A trie also works but needs more code."
 ],
 "best": "Sort once; bisect to the first match and take up to 3.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def suggestedProducts(self, products, searchWord):\n        products.sort()\n        res = []\n        for i in range(1, len(searchWord) + 1):\n            prefix = searchWord[:i]\n            k = bisect_left(products, prefix)\n            res.append([w for w in products[k:k+3] if w.startswith(prefix)])\n        return res",
   "time": "O(n log n + L log n)",
   "space": "O(1) extra",
   "explain": "Sorted list + binary search replaces an explicit trie.",
   "frames": [
    {
     "title": "Sorted",
     "description": "mobile, moneypot, monitor, mouse, mousepad",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "products",
       "cells": [
        "mobile",
        "moneypot",
        "monitor",
        "mouse",
        "mousepad"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Typed 'm'",
     "description": "First match at index 0 → take up to 3 that start with 'm'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "products",
       "cells": [
        "mobile",
        "moneypot",
        "monitor",
        "mouse",
        "mousepad"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {
        "0": "k"
       }
      },
      {
       "name": "suggest",
       "cells": [
        "mobile",
        "moneypot",
        "monitor"
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
     "title": "Typed 'mo'",
     "description": "First match at index 0 → take up to 3 that start with 'mo'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "products",
       "cells": [
        "mobile",
        "moneypot",
        "monitor",
        "mouse",
        "mousepad"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {
        "0": "k"
       }
      },
      {
       "name": "suggest",
       "cells": [
        "mobile",
        "moneypot",
        "monitor"
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
     "title": "Typed 'mou'",
     "description": "First match at index 3 → take up to 3 that start with 'mou'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "products",
       "cells": [
        "mobile",
        "moneypot",
        "monitor",
        "mouse",
        "mousepad"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {
        "3": "k"
       }
      },
      {
       "name": "suggest",
       "cells": [
        "mouse",
        "mousepad"
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
     "title": "Typed 'mous'",
     "description": "First match at index 3 → take up to 3 that start with 'mous'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "products",
       "cells": [
        "mobile",
        "moneypot",
        "monitor",
        "mouse",
        "mousepad"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {
        "3": "k"
       }
      },
      {
       "name": "suggest",
       "cells": [
        "mouse",
        "mousepad"
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
     "title": "Typed 'mouse'",
     "description": "First match at index 3 → take up to 3 that start with 'mouse'.",
     "codeLines": [
      7,
      8
     ],
     "facts": {},
     "rows": [
      {
       "name": "products",
       "cells": [
        "mobile",
        "moneypot",
        "monitor",
        "mouse",
        "mousepad"
       ],
       "hot": [
        3,
        4
       ],
       "marks": {
        "3": "k"
       }
      },
      {
       "name": "suggest",
       "cells": [
        "mouse",
        "mousepad"
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
