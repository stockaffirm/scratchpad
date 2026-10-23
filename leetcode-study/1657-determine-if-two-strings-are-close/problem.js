window.PROBLEM = {
 "id": "1657",
 "title": "Determine if Two Strings Are Close",
 "level": "Medium",
 "topic": "Hash Map / Set",
 "question": "Two moves are allowed: swap any two letters' positions, or swap two letter types everywhere (every a becomes b and every b becomes a). Can word1 become word2?",
 "input": "\"cabbba\", \"abbccc\"",
 "output": "True",
 "notes": [
  "Swapping positions = order is free.",
  "Swapping identities = only the frequency multiset matters."
 ],
 "best": "Two checks: same letter set, same sorted frequency list.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def closeStrings(self, w1, w2):\n        c1, c2 = Counter(w1), Counter(w2)\n        return set(c1) == set(c2) and sorted(c1.values()) == sorted(c2.values())",
   "time": "O(n log n)",
   "space": "O(1)",
   "explain": "Same letters, and same multiset of frequencies.",
   "frames": [
    {
     "title": "Input",
     "description": "Two words.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "w1",
       "cells": [
        "c",
        "a",
        "b",
        "b",
        "b",
        "a"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "w2",
       "cells": [
        "a",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Counts",
     "description": "w1: a2 b3 c1 · w2: a1 b2 c3",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "w1 letters",
       "cells": [
        "a:2",
        "b:3",
        "c:1"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "w2 letters",
       "cells": [
        "a:1",
        "b:2",
        "c:3"
       ],
       "hot": [],
       "marks": {}
      }
     ],
     "sub": [
      {
       "description": "w1: see 'c' → c:1",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1",
         "cells": [
          "c",
          "a",
          "b",
          "b",
          "b",
          "a"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "w1 counts",
         "cells": [
          "c:1"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w1: see 'a' → c:1 a:1",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1",
         "cells": [
          "c",
          "a",
          "b",
          "b",
          "b",
          "a"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:1"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w1: see 'b' → c:1 a:1 b:1",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1",
         "cells": [
          "c",
          "a",
          "b",
          "b",
          "b",
          "a"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:1",
          "b:1"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w1: see 'b' → c:1 a:1 b:2",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1",
         "cells": [
          "c",
          "a",
          "b",
          "b",
          "b",
          "a"
         ],
         "hot": [
          3
         ],
         "marks": {}
        },
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:1",
          "b:2"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w1: see 'b' → c:1 a:1 b:3",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1",
         "cells": [
          "c",
          "a",
          "b",
          "b",
          "b",
          "a"
         ],
         "hot": [
          4
         ],
         "marks": {}
        },
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:1",
          "b:3"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w1: see 'a' → c:1 a:2 b:3",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1",
         "cells": [
          "c",
          "a",
          "b",
          "b",
          "b",
          "a"
         ],
         "hot": [
          5
         ],
         "marks": {}
        },
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w2: see 'a' → a:1",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "w2",
         "cells": [
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          0
         ],
         "marks": {}
        },
        {
         "name": "w2 counts",
         "cells": [
          "a:1"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w2: see 'b' → a:1 b:1",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "w2",
         "cells": [
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          1
         ],
         "marks": {}
        },
        {
         "name": "w2 counts",
         "cells": [
          "a:1",
          "b:1"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w2: see 'b' → a:1 b:2",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "w2",
         "cells": [
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          2
         ],
         "marks": {}
        },
        {
         "name": "w2 counts",
         "cells": [
          "a:1",
          "b:2"
         ],
         "hot": [
          1
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w2: see 'c' → a:1 b:2 c:1",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "w2",
         "cells": [
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          3
         ],
         "marks": {}
        },
        {
         "name": "w2 counts",
         "cells": [
          "a:1",
          "b:2",
          "c:1"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w2: see 'c' → a:1 b:2 c:2",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "w2",
         "cells": [
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          4
         ],
         "marks": {}
        },
        {
         "name": "w2 counts",
         "cells": [
          "a:1",
          "b:2",
          "c:2"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "w2: see 'c' → a:1 b:2 c:3",
       "codeLines": [
        3
       ],
       "facts": {},
       "rows": [
        {
         "name": "w1 counts",
         "cells": [
          "c:1",
          "a:2",
          "b:3"
         ],
         "hot": [],
         "marks": {}
        },
        {
         "name": "w2",
         "cells": [
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          5
         ],
         "marks": {}
        },
        {
         "name": "w2 counts",
         "cells": [
          "a:1",
          "b:2",
          "c:3"
         ],
         "hot": [
          2
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Same letters?",
     "description": "Both use {a, b, c}.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "set(c1)",
       "cells": [
        "a",
        "b",
        "c"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "set(c2)",
       "cells": [
        "a",
        "b",
        "c"
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
     "title": "Same frequencies?",
     "description": "Sorted: [1,2,3] vs [1,2,3].",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "sorted c1",
       "cells": [
        "1",
        "2",
        "3"
       ],
       "hot": [
        0,
        1,
        2
       ],
       "marks": {}
      },
      {
       "name": "sorted c2",
       "cells": [
        "1",
        "2",
        "3"
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
     "description": "Both checks pass.",
     "codeLines": [
      4
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
 ],
 "why": "cabbba has a×2, b×3, c×1. abbccc has a×1, b×2, c×3. Same letters, and the counts {1,2,3} match → True.",
 "idea": "They need the same set of letters and the same list of counts (in any order)."
};
