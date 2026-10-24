# LeetCode Visual Study

Open `index.html`. Each problem: **Question → Code → Flow → Diagram** with Prev / Play / Next (also ← → and space).

```
leetcode-study/
├── index.html            landing-page index: LeetCode 75 and LeetCode 150 lists, expandable question previews
├── assets/               shared style, player, progress, toc scripts
└── 238-product-except-self/   one folder per problem
    ├── index.html
    └── problem.js        question, code, steps (edit to change a lesson)
```

To add a problem: copy a folder, edit `problem.js`, add its folder name to `assets/order.js` and a card to `index.html`.
