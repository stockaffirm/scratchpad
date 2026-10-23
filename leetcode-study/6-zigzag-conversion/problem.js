window.PROBLEM = {
 "id": "6",
 "title": "Zigzag Conversion",
 "level": "Medium",
 "topic": "Array / String",
 "question": "Write the text in a zigzag down and up over numRows rows, then read it row by row.",
 "why": "PAYPALISHIRING on 3 rows reads PAHNAPLSIIGYIR.",
 "idea": "Walk through the letters, moving the row down then up, bouncing at the top and bottom.",
 "input": "\"PAYPALISHIRING\", numRows=3",
 "output": "\"PAHNAPLSIIGYIR\"",
 "notes": [],
 "best": "Walk through the letters, moving the row down then up, bouncing at the top and bottom.",
 "variants": [
  {
   "name": "Solution",
   "code": "class Solution:\n    def convert(self, s, numRows):\n        if numRows == 1: return s\n        rows, r, d = [''] * numRows, 0, 1\n        for ch in s:\n            rows[r] += ch\n            if r == 0: d = 1\n            elif r == numRows - 1: d = -1\n            r += d\n        return ''.join(rows)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Walk through the letters, moving the row down then up, bouncing at the top and bottom.",
   "frames": [
    {
     "title": "Start",
     "description": "3 empty rows; direction starts down.",
     "codeLines": [
      4
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'P'",
     "description": "Place in row 0.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'A'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A"
       ],
       "hot": [
        0
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "·"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'Y'",
     "description": "Place in row 2.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y"
       ],
       "hot": [
        0
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'P'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'A'",
     "description": "Place in row 0.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A"
       ],
       "hot": [
        1
       ],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'L'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'I'",
     "description": "Place in row 2.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I"
       ],
       "hot": [
        1
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'S'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'H'",
     "description": "Place in row 0.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A",
        "H"
       ],
       "hot": [
        2
       ],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'I'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A",
        "H"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S",
        "I"
       ],
       "hot": [
        4
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'R'",
     "description": "Place in row 2.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A",
        "H"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S",
        "I"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I",
        "R"
       ],
       "hot": [
        2
       ],
       "marks": {}
      }
     ]
    },
    {
     "title": "'I'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A",
        "H"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S",
        "I",
        "I"
       ],
       "hot": [
        5
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I",
        "R"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'N'",
     "description": "Place in row 0.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A",
        "H",
        "N"
       ],
       "hot": [
        3
       ],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S",
        "I",
        "I"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I",
        "R"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "'G'",
     "description": "Place in row 1.",
     "codeLines": [
      6,
      9
     ],
     "facts": {},
     "rows": [
      {
       "name": "row 0",
       "cells": [
        "P",
        "A",
        "H",
        "N"
       ],
       "hot": [],
       "marks": {}
      },
      {
       "name": "row 1",
       "cells": [
        "A",
        "P",
        "L",
        "S",
        "I",
        "I",
        "G"
       ],
       "hot": [
        6
       ],
       "marks": {}
      },
      {
       "name": "row 2",
       "cells": [
        "Y",
        "I",
        "R"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    },
    {
     "title": "Return",
     "description": "Join the rows.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "P",
        "A",
        "H",
        "N",
        "A",
        "P",
        "L",
        "S",
        "I",
        "I",
        "G",
        "Y",
        "I",
        "R"
       ],
       "hot": [],
       "marks": {}
      }
     ]
    }
   ]
  }
 ]
};
