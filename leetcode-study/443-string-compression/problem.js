window.PROBLEM = {
 "id": "443",
 "title": "String Compression",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Shrink a list of letters in place. Replace each run of the same letter with the letter and how many times it repeated (skip the number if it is 1). Return the new length.",
 "input": "[\"a\",\"a\",\"b\",\"b\",\"c\",\"c\",\"c\"]",
 "output": "6 → [\"a\",\"2\",\"b\",\"2\",\"c\",\"3\"]",
 "notes": [
  "write never passes read, so overwriting is safe.",
  "Counts above 9 are written digit by digit."
 ],
 "best": "Read/write pointers compress in place with O(1) extra space.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def compress(self, chars):\n        write = read = 0\n        while read < len(chars):\n            ch, count = chars[read], 0\n            while read < len(chars) and chars[read] == ch:\n                read += 1\n                count += 1\n            chars[write] = ch\n            write += 1\n            if count > 1:\n                for d in str(count):\n                    chars[write] = d\n                    write += 1\n        return write",
   "time": "O(n)",
   "space": "O(1)",
   "explain": "Read pointer scans a run; write pointer lays down the compressed form.",
   "frames": [
    {
     "title": "Input",
     "description": "Runs: aa, bb, ccc.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [],
       "marks": {
        "0": "read,write"
       }
      }
     ]
    },
    {
     "title": "Run of a",
     "description": "Read 2 × 'a'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "count": 2
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "0": "write",
        "2": "read"
       }
      }
     ],
     "sub": [
      {
       "description": "ch = 'a', count = 0 — start counting this run",
       "codeLines": [
        5
       ],
       "facts": {
        "count": 0
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [],
         "marks": {
          "0": "read,write"
         }
        }
       ]
      },
      {
       "description": "chars[0] is 'a' = ch → read moves to 1, count = 1",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 1
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
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
         "marks": {
          "0": "write",
          "1": "read"
         }
        }
       ]
      },
      {
       "description": "chars[1] is 'a' = ch → read moves to 2, count = 2",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 2
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          0,
          1
         ],
         "marks": {
          "0": "write",
          "2": "read"
         }
        }
       ]
      },
      {
       "description": "chars[2] is 'b' ≠ 'a' → the run ends. Count = 2",
       "codeLines": [
        6
       ],
       "facts": {
        "count": 2
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "a",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          0,
          1
         ],
         "marks": {
          "0": "write",
          "2": "read"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Write a2",
     "description": "Write letter and count at the write pointer.",
     "codeLines": [
      9,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [
        0,
        1
       ],
       "marks": {
        "2": "read,write"
       }
      }
     ],
     "sub": [
      {
       "description": "Write the letter 'a' at position 0 (write → 1)",
       "codeLines": [
        9,
        10
       ],
       "facts": {},
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
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
         "marks": {
          "1": "write",
          "2": "read"
         }
        }
       ]
      },
      {
       "description": "count 2 > 1 → write the digit '2' at position 1 (write → 2)",
       "codeLines": [
        12,
        13,
        14
       ],
       "facts": {},
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          1
         ],
         "marks": {
          "2": "read,write"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Run of b",
     "description": "Read 2 × 'b'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "count": 2
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "b",
        "c",
        "c",
        "c"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {
        "2": "write",
        "4": "read"
       }
      }
     ],
     "sub": [
      {
       "description": "ch = 'b', count = 0 — start counting this run",
       "codeLines": [
        5
       ],
       "facts": {
        "count": 0
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [],
         "marks": {
          "2": "read,write"
         }
        }
       ]
      },
      {
       "description": "chars[2] is 'b' = ch → read moves to 3, count = 1",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 1
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          2
         ],
         "marks": {
          "2": "write",
          "3": "read"
         }
        }
       ]
      },
      {
       "description": "chars[3] is 'b' = ch → read moves to 4, count = 2",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 2
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          2,
          3
         ],
         "marks": {
          "2": "write",
          "4": "read"
         }
        }
       ]
      },
      {
       "description": "chars[4] is 'c' ≠ 'b' → the run ends. Count = 2",
       "codeLines": [
        6
       ],
       "facts": {
        "count": 2
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          2,
          3
         ],
         "marks": {
          "2": "write",
          "4": "read"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Write b2",
     "description": "Write letter and count at the write pointer.",
     "codeLines": [
      9,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "c",
        "c"
       ],
       "hot": [
        2,
        3
       ],
       "marks": {
        "4": "read,write"
       }
      }
     ],
     "sub": [
      {
       "description": "Write the letter 'b' at position 2 (write → 3)",
       "codeLines": [
        9,
        10
       ],
       "facts": {},
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "b",
          "c",
          "c",
          "c"
         ],
         "hot": [
          2
         ],
         "marks": {
          "3": "write",
          "4": "read"
         }
        }
       ]
      },
      {
       "description": "count 2 > 1 → write the digit '2' at position 3 (write → 4)",
       "codeLines": [
        12,
        13,
        14
       ],
       "facts": {},
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [
          3
         ],
         "marks": {
          "4": "read,write"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Run of c",
     "description": "Read 3 × 'c'.",
     "codeLines": [
      5,
      6
     ],
     "facts": {
      "count": 3
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "c",
        "c"
       ],
       "hot": [
        4,
        5,
        6
       ],
       "marks": {
        "4": "write",
        "7": "read"
       }
      }
     ],
     "sub": [
      {
       "description": "ch = 'c', count = 0 — start counting this run",
       "codeLines": [
        5
       ],
       "facts": {
        "count": 0
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [],
         "marks": {
          "4": "read,write"
         }
        }
       ]
      },
      {
       "description": "chars[4] is 'c' = ch → read moves to 5, count = 1",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 1
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [
          4
         ],
         "marks": {
          "4": "write",
          "5": "read"
         }
        }
       ]
      },
      {
       "description": "chars[5] is 'c' = ch → read moves to 6, count = 2",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 2
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [
          4,
          5
         ],
         "marks": {
          "4": "write",
          "6": "read"
         }
        }
       ]
      },
      {
       "description": "chars[6] is 'c' = ch → read moves to 7, count = 3",
       "codeLines": [
        7,
        8
       ],
       "facts": {
        "count": 3
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [
          4,
          5,
          6
         ],
         "marks": {
          "4": "write",
          "7": "read"
         }
        }
       ]
      },
      {
       "description": "End of the list → the run ends. Count = 3",
       "codeLines": [
        6
       ],
       "facts": {
        "count": 3
       },
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [
          4,
          5,
          6
         ],
         "marks": {
          "4": "write",
          "7": "read"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Write c3",
     "description": "Write letter and count at the write pointer.",
     "codeLines": [
      9,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "3",
        "c"
       ],
       "hot": [
        4,
        5
       ],
       "marks": {
        "6": "write",
        "7": "read"
       }
      }
     ],
     "sub": [
      {
       "description": "Write the letter 'c' at position 4 (write → 5)",
       "codeLines": [
        9,
        10
       ],
       "facts": {},
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "c",
          "c"
         ],
         "hot": [
          4
         ],
         "marks": {
          "5": "write",
          "7": "read"
         }
        }
       ]
      },
      {
       "description": "count 3 > 1 → write the digit '3' at position 5 (write → 6)",
       "codeLines": [
        12,
        13,
        14
       ],
       "facts": {},
       "rows": [
        {
         "name": "chars",
         "cells": [
          "a",
          "2",
          "b",
          "2",
          "c",
          "3",
          "c"
         ],
         "hot": [
          5
         ],
         "marks": {
          "6": "write",
          "7": "read"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Return",
     "description": "Answer is the write position.",
     "codeLines": [
      15
     ],
     "facts": {
      "length": 6
     },
     "rows": [
      {
       "name": "chars",
       "cells": [
        "a",
        "2",
        "b",
        "2",
        "c",
        "3"
       ],
       "hot": [
        0,
        1,
        2,
        3,
        4,
        5
       ],
       "marks": {}
      }
     ]
    }
   ]
  }
 ],
 "why": "a a b b c c c becomes a 2 b 2 c 3. That is 6 characters long.",
 "idea": "One pointer reads ahead to count a run. Another writes the result behind it."
};
