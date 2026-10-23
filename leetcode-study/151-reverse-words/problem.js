window.PROBLEM = {
 "id": "151",
 "title": "Reverse Words in a String",
 "level": "Medium",
 "topic": "Strings",
 "question": "Reverse the order of the words. Exactly one space between words, no spaces at the start or end.",
 "input": "\"  the  sky is blue  \"",
 "output": "\"blue is sky the\"",
 "notes": [
  "Manual algorithm: normalize spaces, reverse the whole character list, then reverse each word.",
  "deque.appendleft(word) adds each new word to the front.",
  "split() without parameters removes redundant whitespace automatically.",
  "Built-in version is the most concise; manual is best for showing algorithm fundamentals."
 ],
 "best": "For production Python, split + reversed + join is clearest. For an interview without built-ins, explain the manual double-reversal method.",
 "variants": [
  {
   "name": "Manual: reverse twice",
   "code": "class Solution:\n    def trim_spaces(self, s):\n        a = ' '.join(s.split())\n        return list(a)\n    def reverse(self, a, i, j):\n        while i < j:\n            a[i], a[j] = a[j], a[i]\n            i, j = i+1, j-1\n    def reverseWords(self, s):\n        a = self.trim_spaces(s)\n        self.reverse(a, 0, len(a)-1)\n        start = 0\n        for end in range(len(a)+1):\n            if end == len(a) or a[end] == ' ':\n                self.reverse(a, start, end-1)\n                start = end+1\n        return ''.join(a)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Shows how reversing everything and then each word changes order without changing word spelling.",
   "frames": [
    {
     "title": "Input with spaces",
     "description": "Example string has leading, trailing and doubled spaces.",
     "codeLines": [
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "input",
       "cells": [
        " ",
        " ",
        "t",
        "h",
        "e",
        " ",
        " ",
        "s",
        "k",
        "y",
        " ",
        "i",
        "s",
        " ",
        "b",
        "l",
        "u",
        "e",
        " ",
        " "
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Trim + compress spaces",
     "description": "After normalization, every word has one space between it.",
     "codeLines": [
      2,
      3,
      4,
      10
     ],
     "facts": {},
     "rows": [
      {
       "name": "normalized",
       "cells": [
        "t",
        "h",
        "e",
        " ",
        "s",
        "k",
        "y",
        " ",
        "i",
        "s",
        " ",
        "b",
        "l",
        "u",
        "e"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Reverse entire sequence",
     "description": "All characters reversed: eulb si yks eht.",
     "codeLines": [
      5,
      6,
      7,
      8,
      11
     ],
     "facts": {},
     "rows": [
      {
       "name": "reversed",
       "cells": [
        "e",
        "u",
        "l",
        "b",
        " ",
        "s",
        "i",
        " ",
        "y",
        "k",
        "s",
        " ",
        "e",
        "h",
        "t"
       ],
       "hot": []
      }
     ],
     "sub": [
      {
       "description": "swap positions 0 and 14",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "h",
          "e",
          " ",
          "s",
          "k",
          "y",
          " ",
          "i",
          "s",
          " ",
          "b",
          "l",
          "u",
          "t"
         ],
         "hot": [
          0,
          14
         ],
         "marks": {
          "0": "i",
          "14": "j"
         }
        }
       ]
      },
      {
       "description": "swap positions 1 and 13",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "u",
          "e",
          " ",
          "s",
          "k",
          "y",
          " ",
          "i",
          "s",
          " ",
          "b",
          "l",
          "h",
          "t"
         ],
         "hot": [
          1,
          13
         ],
         "marks": {
          "1": "i",
          "13": "j"
         }
        }
       ]
      },
      {
       "description": "swap positions 2 and 12",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "u",
          "l",
          " ",
          "s",
          "k",
          "y",
          " ",
          "i",
          "s",
          " ",
          "b",
          "e",
          "h",
          "t"
         ],
         "hot": [
          2,
          12
         ],
         "marks": {
          "2": "i",
          "12": "j"
         }
        }
       ]
      },
      {
       "description": "swap positions 3 and 11",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "u",
          "l",
          "b",
          "s",
          "k",
          "y",
          " ",
          "i",
          "s",
          " ",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          3,
          11
         ],
         "marks": {
          "3": "i",
          "11": "j"
         }
        }
       ]
      },
      {
       "description": "swap positions 4 and 10",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "u",
          "l",
          "b",
          " ",
          "k",
          "y",
          " ",
          "i",
          "s",
          "s",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          4,
          10
         ],
         "marks": {
          "4": "i",
          "10": "j"
         }
        }
       ]
      },
      {
       "description": "swap positions 5 and 9",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "u",
          "l",
          "b",
          " ",
          "s",
          "y",
          " ",
          "i",
          "k",
          "s",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          5,
          9
         ],
         "marks": {
          "5": "i",
          "9": "j"
         }
        }
       ]
      },
      {
       "description": "swap positions 6 and 8",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "e",
          "u",
          "l",
          "b",
          " ",
          "s",
          "i",
          " ",
          "y",
          "k",
          "s",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          6,
          8
         ],
         "marks": {
          "6": "i",
          "8": "j"
         }
        }
       ]
      }
     ]
    },
    {
     "title": "Reverse each individual word",
     "description": "eulb→blue, si→is, yks→sky, eht→the.",
     "codeLines": [
      12,
      13,
      14,
      15,
      16,
      17
     ],
     "facts": {},
     "rows": [
      {
       "name": "result",
       "cells": [
        "b",
        "l",
        "u",
        "e",
        " ",
        "i",
        "s",
        " ",
        "s",
        "k",
        "y",
        " ",
        "t",
        "h",
        "e"
       ],
       "hot": []
      }
     ],
     "sub": [
      {
       "description": "reverse the word at 0–3: swap 0 and 3",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "b",
          "u",
          "l",
          "e",
          " ",
          "s",
          "i",
          " ",
          "y",
          "k",
          "s",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          0,
          3
         ],
         "marks": {
          "0": "i",
          "3": "j"
         }
        }
       ]
      },
      {
       "description": "reverse the word at 0–3: swap 1 and 2",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "b",
          "l",
          "u",
          "e",
          " ",
          "s",
          "i",
          " ",
          "y",
          "k",
          "s",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          1,
          2
         ],
         "marks": {
          "1": "i",
          "2": "j"
         }
        }
       ]
      },
      {
       "description": "reverse the word at 5–6: swap 5 and 6",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "b",
          "l",
          "u",
          "e",
          " ",
          "i",
          "s",
          " ",
          "y",
          "k",
          "s",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          5,
          6
         ],
         "marks": {
          "5": "i",
          "6": "j"
         }
        }
       ]
      },
      {
       "description": "reverse the word at 8–10: swap 8 and 10",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "b",
          "l",
          "u",
          "e",
          " ",
          "i",
          "s",
          " ",
          "s",
          "k",
          "y",
          " ",
          "e",
          "h",
          "t"
         ],
         "hot": [
          8,
          10
         ],
         "marks": {
          "8": "i",
          "10": "j"
         }
        }
       ]
      },
      {
       "description": "reverse the word at 12–14: swap 12 and 14",
       "codeLines": [
        7
       ],
       "facts": {},
       "rows": [
        {
         "name": "a",
         "cells": [
          "b",
          "l",
          "u",
          "e",
          " ",
          "i",
          "s",
          " ",
          "s",
          "k",
          "y",
          " ",
          "t",
          "h",
          "e"
         ],
         "hot": [
          12,
          14
         ],
         "marks": {
          "12": "i",
          "14": "j"
         }
        }
       ]
      }
     ]
    }
   ]
  },
  {
   "name": "Deque appendleft",
   "code": "from collections import deque\nclass Solution:\n    def reverseWords(self, s):\n        d = deque()\n        for word in s.split():\n            d.appendleft(word)\n        return ' '.join(d)",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Deque supports fast appending to the front.",
   "frames": [
    {
     "title": "Input",
     "description": "The words are the, sky, is, blue.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "the",
        "sky",
        "is",
        "blue"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "Append each word at the front",
     "description": "Deque changes as each word arrives.",
     "codeLines": [
      4,
      5,
      6
     ],
     "facts": {},
     "rows": [
      {
       "name": "deque",
       "cells": [
        "blue",
        "is",
        "sky",
        "the"
       ],
       "hot": [
        0
       ]
      }
     ],
     "sub": [
      {
       "description": "appendleft('the') → front of the deque",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "deque",
         "cells": [
          "the"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "appendleft('sky') → front of the deque",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "deque",
         "cells": [
          "sky",
          "the"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "appendleft('is') → front of the deque",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "deque",
         "cells": [
          "is",
          "sky",
          "the"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      },
      {
       "description": "appendleft('blue') → front of the deque",
       "codeLines": [
        6
       ],
       "facts": {},
       "rows": [
        {
         "name": "deque",
         "cells": [
          "blue",
          "is",
          "sky",
          "the"
         ],
         "hot": [
          0
         ],
         "marks": {}
        }
       ]
      }
     ]
    },
    {
     "title": "Join",
     "description": "Join deque words with one space.",
     "codeLines": [
      7
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "b",
        "l",
        "u",
        "e",
        " ",
        "i",
        "s",
        " ",
        "s",
        "k",
        "y",
        " ",
        "t",
        "h",
        "e"
       ],
       "hot": []
      }
     ]
    }
   ]
  },
  {
   "name": "split + reversed + join",
   "code": "class Solution:\n    def reverseWords(self, s):\n        return ' '.join(reversed(s.split()))",
   "time": "O(n)",
   "space": "O(n)",
   "explain": "Short, idiomatic Python solution.",
   "frames": [
    {
     "title": "Input",
     "description": "Leading, trailing, and repeated spaces appear.",
     "codeLines": [
      2
     ],
     "facts": {},
     "rows": [
      {
       "name": "input",
       "cells": [
        " ",
        " ",
        "t",
        "h",
        "e",
        " ",
        " ",
        "s",
        "k",
        "y",
        " ",
        "i",
        "s",
        " ",
        "b",
        "l",
        "u",
        "e",
        " ",
        " "
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "split()",
     "description": "Split into words while removing extra spaces.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "words",
       "cells": [
        "the",
        "sky",
        "is",
        "blue"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "reversed()",
     "description": "Reverse the word order.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "reversed",
       "cells": [
        "blue",
        "is",
        "sky",
        "the"
       ],
       "hot": []
      }
     ]
    },
    {
     "title": "join()",
     "description": "Put exactly one space between words.",
     "codeLines": [
      3
     ],
     "facts": {},
     "rows": [
      {
       "name": "answer",
       "cells": [
        "b",
        "l",
        "u",
        "e",
        " ",
        "i",
        "s",
        " ",
        "s",
        "k",
        "y",
        " ",
        "t",
        "h",
        "e"
       ],
       "hot": []
      }
     ]
    }
   ]
  }
 ],
 "why": "'  the  sky is blue  ' has the words the, sky, is, blue. Reversed → blue is sky the.",
 "idea": "split() already removes extra spaces. Reverse the word list and join with a space."
};
