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

## Tools

`tools/gen/` is the generator pipeline (`./all.sh`): curated walkthroughs + real-code traces + extra approaches for every problem. See `tools/gen/drivers.py` (examples), `drivers2.py` (edge cases) and `alts*.js` (other approaches).

## Done marks

Saved in the browser. **Sign in with GitHub to sync** (index page) stores them in a dedicated Supabase project (`leetcode-study-sync`), one private row per GitHub user (table `leetcode_progress`, row-level security), so they follow you across browsers. Changes merge per problem by timestamp, so a fresh browser never wipes saved progress.

## Adding a set (e.g. Google)

Use the `leetcode-study-add` skill (in `stockaffirm/truistAI`): `tools/gen/delta.js` finds the problems not yet on the site, `tools/gen/sets.json` registers the set (new section with only the new problems + a highlight button).
