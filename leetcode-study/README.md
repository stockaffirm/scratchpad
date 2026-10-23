# LeetCode Visual Study Studio

Interactive study material from our ChatGPT LeetCode sessions.

**Open `leetcode-study/index.html` in a browser** (or serve the folder locally with `python3 -m http.server 8000` and visit `http://localhost:8000/leetcode-study/`).

## Features

- Previous / Next / Play / Pause / Reset controls; adjustable speed and timeline scrubber
- Top-to-bottom illustrated algorithm execution using example input
- Code snippets with the currently relevant lines highlighted
- Separate walkthroughs for multiple solutions and recommendations on which is better
- Time and space complexity for each solution
- Highlighted notes and conceptual review takeaways from our conversation
- Personal notes saved in the current browser via localStorage; optional Markdown export

## Included problems

| LeetCode | Problem | Variants |
| --- | --- | --- |
| 1071 | Greatest Common Divisor of Strings | candidate scan; concat + GCD |
| 1431 | Kids With the Greatest Number of Candies | built-in max; manual max |
| 605 | Can Place Flowers | greedy + early return |
| 345 | Reverse Vowels of a String | two pointers |
| 151 | Reverse Words in a String | manual reversal; deque; split/reversed/join |
| 238 | Product of Array Except Self | left and right arrays; running right product |
| 334 | Increasing Triplet Subsequence | greedy thresholds |

## Project files

- `index.html` — complete responsive interactive page.
- `lessons.js` — Python solutions, animation frames, complexity notes, and review highlights.

## Notes

- Animations are **step-through visualizations**, not executed Python programs. Each step illustrates an example execution.
- The three Reverse Words solutions are represented with pedagogically equivalent code. The manual variant's trim helper uses Python split to keep the demo compact rather than reproduce every original helper line.
- Personal notes are not automatically committed back to GitHub; click **Export Markdown** if you want a file you can version-control.
- For this conversation, the optional `n == 0` guard is included in the Can Place Flowers demonstration.
