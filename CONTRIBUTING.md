# Contributing

There are two ways to contribute: **reproduction logs** (the normal path for students) and
**fixes/improvements** (docs, code, new lessons).

## Reproduction logs

This is the normal path for students. The workflow is borrowed from
[castorini/onboarding](https://github.com/castorini/onboarding). Follow every step in order —
the Git commands are spelled out so you can copy-paste them even if you've never used Git before.

### Step 1 — Fork and clone your copy

If you haven't already (the [README](README.md#how-to-participate) covers this):

1. Click **Fork** on `https://github.com/abubinfahd/ml-foundations` to make your own copy.
2. Clone _your fork_ and move into it (replace `<your-username>`):

   ```bash
   git clone https://github.com/<your-username>/ml-foundations.git
   cd ml-foundations
   ```

### Step 2 — Connect to the original repo (upstream)

This lets you pull in updates the maintainers make later:

```bash
git remote add upstream https://github.com/abubinfahd/ml-foundations.git
git pull upstream master
```

You only need `git remote add upstream …` **once**. Run `git pull upstream master` any time you
want the latest changes before starting new work.

### Step 3 — Reproduce a lesson

Work through a lesson **exactly as written** — same commands, same environment. Your printed
output must match the _Expected output_ blocks in the doc.

### Step 4 — Add your reproduction-log line

Open the lesson doc and scroll to the **Reproduction Log** section at the bottom. Add **one line**
_below_ any existing lines (keep them in date order):

```
+ Results reproduced by [@abubinfahd](https://github.com/abubinfahd) on 2026-07-14 (commit [abc1234](https://github.com/abubinfahd/ml-foundations/commit/abc1234567890abcdef1234567890abcdef1234))
```

Fill in each part:

- **Username**: replace `abubinfahd` with your own GitHub username (in both the `@…` text and the
  link).
- **Date**: the day you reproduced it, formatted `YYYY-MM-DD`.
- **Commit**: must be on the `master` branch of `ml-foundations`. Get the two hashes with:

  ```bash
  git rev-parse --short HEAD   # short hash → the [abc1234] anchor text
  git rev-parse HEAD           # full hash  → the end of the commit URL
  ```

  The full URL looks like
  `https://github.com/abubinfahd/ml-foundations/commit/<full-hash>`.

### Step 5 — Commit, push, and open a pull request

```bash
git add docs/                                   # stage the lesson docs you edited
git commit -m "Add reproduction logs for lessons 01-03"
git push origin master                          # push to your fork
```

Then, on GitHub:

1. Open your fork's page — GitHub shows a **"Compare & pull request"** button. Click it.
2. Make sure the PR targets `abubinfahd/ml-foundations`'s `master` branch.
3. Fill in the PR template (see the example below) and submit.

### PR rules

- **One consolidated PR** per batch of lessons — do _not_ open a separate PR for each
  lesson or each file. Finish a few lessons, then send one PR with all your log lines.
- The PR must touch **only** Reproduction Log lines. No other edits mixed in.
- Fill in the PR template checklist honestly. Maintainers may ask you to paste your
  terminal output.
- In the PR description, note your setup (OS, environment/configuration) and whether
  everything worked or you hit issues — and mention anything you think the lesson doc
  could clarify.

### Example PR description

A filled-in reproduction-log PR for lessons 01–03, using the
[PR template](.github/PULL_REQUEST_TEMPLATE.md):

```markdown
## What this PR does
Adds reproduction log entries for lessons 01-03.

## Checklist (reproduction-log PRs)
- [x] Ran every command myself, in order, in PowerShell on Windows.
- [x] Output matched the *Expected output* blocks (metrics within the usual last-decimal wobble).
- [x] Only Reproduction Log lines changed.
- [x] Log line format correct, commit is on `master`.
- [x] Single consolidated PR.

## Environment
- OS: Windows 11
- Python 3.12.3, `.venv`, packages pinned per `requirements.txt`

## Result
- Success: everything matched for lessons 01-03.
- Issues: none.

## Suggestions (optional)
Lesson 02's CLT section could clarify why n=30 was chosen.
```

## Fixes and improvements

Bug fixes, clearer explanations, and new lessons are welcome — but keep them in
**separate PRs** from reproduction logs.

- For anything non-trivial, [open an issue](../../issues/new/choose) first to discuss.
- If a script's output changes, update the _Expected output_ blocks in the lesson doc **and**
  the matching file in `tests/expected/` in the same PR (see below), and note that existing
  reproduction logs predate the change.
- Match the existing lesson template: Concept → Dataset → From scratch → With a library →
  Check your understanding → Reproduction Log.

### Running the test suite

`tests/test_reproducibility.py` runs every lesson script and checks its stdout against the
`Expected output` block copied into `tests/expected/`. Run it yourself before opening a PR:

```powershell
pip install -r requirements-dev.txt
.\scripts\download_data.ps1 all
python -m pytest tests/ -v
```

The check is tolerant of tiny floating-point differences: the text of each line must match
exactly, but numbers only have to agree within a small tolerance (see the top of
`tests/test_reproducibility.py`). This keeps the neural-network / deep-learning lessons
passing across different CPUs, whose last decimals legitimately differ.

If you intentionally changed a script's output, update both the lesson doc's `Expected
output` block and the corresponding `tests/expected/<name>.txt` file so they stay in sync.

## Reporting problems

Use the issue templates:

- **Bug report** — a command fails, a checksum mismatches, or your output differs from the
  doc. Include your OS, Python version, and the full terminal output.
- **Question** — something in a lesson is unclear. Questions are contributions too: they
  tell us where the docs need work.
