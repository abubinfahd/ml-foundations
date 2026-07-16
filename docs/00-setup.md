# Lesson 00 — Setup

Goal: a working, verified environment. Every later lesson assumes you completed this one.

This lesson is written for **complete beginners to command-line tools** — if you've never
used Git, a terminal, or Python before, read every note (the `>` blockquotes), not just the
commands. If you've done this kind of setup before, skim the numbered steps and skip the
notes.

This course targets **Windows 10/11**. Every command is written for **PowerShell**, the
terminal built into Windows — open it from the Start menu (search "PowerShell" and click
*Windows PowerShell*), or use the integrated terminal in VS Code (menu **Terminal → New
Terminal**, then make sure the dropdown in that panel says "PowerShell", not "cmd" or
"bash").

> **Why a terminal at all?** Every lesson in this course works by running a script and
> reading what it prints, rather than clicking through a GUI. That's deliberate: it's the
> same workflow you'll use for real data science and ML work, and it makes every result
> exactly reproducible — anyone who runs the same command sees the same text. If a command
> below feels unfamiliar, that's normal; you'll be comfortable with all of them by the end
> of this lesson.

## 1. Fork and clone

**Fork** means "make your own copy of this repository on GitHub, under your own account."
You need a copy of your own because you can't push changes directly into someone else's
repository — you push into your fork, then ask (via a *pull request*) for your changes to
be pulled into the original.

1. Make sure you have a free [GitHub account](https://github.com/join) and are signed in.
2. Open `https://github.com/abubinfahd/ml-foundations` in your browser.
3. Click the **Fork** button near the top-right of the page, then **Create fork**. GitHub
   creates `https://github.com/<your-username>/ml-foundations` — this is now yours to push
   to.

**Clone** means "download that repository from GitHub onto your own computer." Open
PowerShell and run, replacing `<your-username>` with your actual GitHub username:

```powershell
git clone https://github.com/<your-username>/ml-foundations.git
cd ml-foundations
git remote add upstream https://github.com/abubinfahd/ml-foundations.git
```

What each line does:

- `git clone <url>` copies the entire repository — every file and its full history — into a
  new folder named `ml-foundations` in whatever directory you were in.
- `cd ml-foundations` moves your terminal *into* that new folder ("cd" = "change
  directory"). **Every command in every lesson from here on assumes you are inside this
  folder** — it's called the "repo root." If a command fails with something like "file not
  found," the first thing to check is `pwd` (prints your current folder) to confirm you're
  in the right place.
- `git remote add upstream …` doesn't download anything — it just teaches your local copy a
  second nickname, `upstream`, pointing at the *original* repo (not your fork). You'll use
  this later if the original repo gets updated and you want those updates. Your own fork is
  already known by the nickname `origin` (git set that up automatically during `clone`).

> **Don't have Git yet?** Install [Git for Windows](https://git-scm.com/download/win) first
> (accept the defaults during install), then reopen PowerShell so it picks up the new
> `git` command, and retry the steps above.

## 2. Check system prerequisites

You need three things: Windows 10/11 (you have this already), Python, and Git (installed in
step 1). Install [Python ≥ 3.10](https://www.python.org/downloads/windows/) if you haven't —
during the installer, **tick the checkbox "Add python.exe to PATH"** on the very first
screen. This matters: without it, typing `python` in a terminal won't find the program you
just installed.

Confirm both are on your PATH (i.e., Windows can find them from any folder) by asking each
program to print its own version number:

```powershell
python --version
git --version
```

You should see something like `Python 3.12.3` and `git version 2.45.0` — the exact numbers
don't matter, only that a version prints instead of an error.

> **"'python' is not recognized as the name of a cmdlet..."** — Python isn't on PATH.
> Reinstall it with **"Add python.exe to PATH"** ticked, then open a **brand-new**
> PowerShell window (existing windows don't pick up PATH changes).
>
> **`python --version` opens the Microsoft Store instead of printing a version** — Windows
> ships a placeholder "python" command that redirects to the Store if no real Python is
> installed, and it can shadow a real install too. Turn it off under *Settings → Apps →
> Advanced app settings → App execution aliases* (toggle "App Installer python.exe" and
> "App Installer python3.exe" off), then reopen PowerShell.
>
> **Multiple Pythons installed?** If you have Python from the Microsoft Store, Anaconda, and
> python.org all on the machine, `python --version` might not be the one you expect. Run
> `where.exe python` to see every `python.exe` PowerShell can find, in the order it will try
> them.

## 3. Create the virtual environment

A **virtual environment** ("venv") is a private, isolated folder of Python packages just for
this project — separate from any other Python projects on your machine. Without one, every
project's package installs land in the same global location, and two projects that need
different versions of the same package (say, one needs pandas 1.x and another needs pandas
3.x) would conflict. A venv sidesteps that entirely: this project gets its own `pandas`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

What each line does:

- `python -m venv .venv` creates that private package folder, named `.venv`, inside your
  repo. (The leading dot just makes it a hidden folder by Unix convention; Git already
  ignores it, so it will never be committed.) This only needs to happen **once** per clone.
- `.venv\Scripts\Activate.ps1` "activates" it: for the rest of this terminal session, typing
  `python` or `pip` uses the venv's private copies instead of your system-wide Python. You'll
  know it worked because your prompt gains a `(.venv)` prefix, e.g. `(.venv) PS
  C:\...\ml-foundations>`.
- `pip install -r requirements.txt` reads the [`requirements.txt`](../requirements.txt) file
  in the repo root — a plain list of package names and exact version numbers — and installs
  every one of them into the now-active venv. This installs pinned versions of NumPy,
  pandas, SciPy, scikit-learn, statsmodels, matplotlib, and **CPU-only** PyTorch (no CUDA
  download — the whole install is well under 1 GB). It takes a few minutes; you'll see a
  scrolling list of `Collecting ...` / `Installing ...` lines while it works.

> **Why pinned exact versions (`numpy==2.5.1`, not just `numpy`)?** So that every learner's
> `pip install` produces byte-for-byte the same library versions, which is what makes the
> *Expected output* blocks throughout this course reliable — a different NumPy version could
> legitimately print a different digit in the 4th decimal place, and pinning removes that
> variable entirely.

> **Activation blocked with an error mentioning "execution policy"?** By default, Windows
> refuses to run *any* PowerShell script, including the venv's own `Activate.ps1`, as a
> security precaution. Allow signed, locally-created scripts for your user account only
> (this does not weaken security for scripts from the internet):
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> Answer "Y" if prompted, then re-run `.venv\Scripts\Activate.ps1`.

> **Every lesson assumes the venv is active.** Activation only lasts for the current
> terminal window/session. Each time you open a **new** PowerShell window to work on a
> lesson, `cd` into the repo and re-run `.venv\Scripts\Activate.ps1` — check for the
> `(.venv)` prefix before running any Python command. You do **not** need to re-run `pip
> install` every time, only re-activate.

## 4. Verify

Run the repo's environment checker:

```powershell
python scripts/verify_setup.py
```

This script ([`scripts/verify_setup.py`](../scripts/verify_setup.py)) does three checks, in
order, and prints one `[OK ]` or `[FAIL]` line per check: it confirms your Python version is
at least 3.10, estimates your total RAM and confirms it's at least ~8 GB, then tries to
`import` each required package (numpy, pandas, scipy, sklearn, statsmodels, matplotlib,
torch) and prints the version it finds. If any package fails to import, that means step 3's
`pip install` didn't complete for that package — re-run `pip install -r requirements.txt`
with the venv active and watch for red error text partway through the scrolling output.

Expected output (versions must match, since they are pinned):

```text
[OK  ] Python 3.x.y (need >= 3.10)
[OK  ] RAM x.x GB (need >= 8 GB)
[OK  ] numpy 2.5.1
[OK  ] pandas 3.0.3
[OK  ] scipy 1.18.0
[OK  ] sklearn 1.9.0
[OK  ] statsmodels 0.14.6
[OK  ] matplotlib 3.11.0
[OK  ] torch 2.13.0+cpu

Setup OK
```

`3.x.y` and `x.x` are placeholders — your Python patch version and exact RAM reading will
differ machine to machine; only the version *numbers of the packages* must match exactly,
since those are pinned in `requirements.txt`. If you see `Setup INCOMPLETE` instead of
`Setup OK` at the end, at least one check above printed `[FAIL]` — fix that one issue and
re-run the script; it's safe to run as many times as you like.

## 5. How every lesson works

Each lesson doc has the same shape, and it's worth knowing what to expect from each part
before you dive into Lesson 01:

1. **Concept** — the theory, with the math, explained in plain language before (or right
   after) each formula. Read this first, even if you're tempted to skip to the code — the
   code is short specifically *because* it assumes you know what it's computing.
2. **Dataset** — one command to download the data for that lesson, with a SHA-256 checksum
   verified automatically so you know the file wasn't corrupted or swapped.
3. **From scratch** — a short Python script, using only NumPy/the standard library, that
   implements the lesson's concept directly from the formulas above. Each lesson doc now
   walks through this script piece by piece — you shouldn't need to guess what any line
   does.
4. **With a library** — the same computation, done with a standard library (pandas,
   scikit-learn, statsmodels, SciPy, or PyTorch). The numbers must match the from-scratch
   run — that agreement is the proof that both you and the library computed the same
   thing.
5. **Check your understanding** — a handful of questions with no answer key. If you can't
   answer one, that's a signal to reread the Concept section or experiment with the script
   (try changing a value and see what happens) before moving on.
6. **Reproduction Log** — a running list of contributors who ran the lesson and got the
   expected output, which you'll add yourself once you're done (see below).

**A typical lesson, start to finish**, once your venv is active:

```powershell
.\scripts\download_data.ps1 <dataset-name>       # step 2 in the lesson doc
python src\lessonNN_xxx\from_scratch.py           # step 3 — read the code first, then run it
python src\lessonNN_xxx\with_library.py           # step 4 — compare the output to step 3
```

Read the printed output next to the *Expected output* block in the doc. They should match
(the doc explains, per lesson, exactly how much numeric wobble — if any — is normal). If
they don't match at all, re-check that your venv is active and that the dataset download
finished without errors before opening an issue.

When your outputs match the doc, append one line to the lesson's Reproduction Log:

```
+ Results reproduced by [@abubinfahd](https://github.com/abubinfahd) on 2026-07-14 (commit [abc1234](https://github.com/abubinfahd/ml-foundations/commit/abc1234567890abcdef1234567890abcdef1234))
```

The link anchor text is the short hash from `git rev-parse --short HEAD`; the link target
is the full hash from `git rev-parse HEAD`. Batch several lessons into a **single PR** —
see [CONTRIBUTING.md](../CONTRIBUTING.md).

From here, you're now ready to proceed to try and reproduce
[Lesson 01 — Descriptive Statistics](01-descriptive-stats.md).

Before you move on, however, add an entry in the "Reproduction Log" at the bottom of this
page, following the same format: use `yyyy-mm-dd`, make sure you're using a commit id
that's on the master branch of `ml-foundations`, and use its 7-hexadecimal prefix for the
link anchor text. In the description of your pull request, please provide some details on
your setup (e.g., operating system, environment and configuration, etc.). In addition,
also provide some indication of success (e.g., everything worked) or document issues you
encountered. If you think this guide can be improved in any way (e.g., you caught a typo
or think a clarification is warranted), feel free to include it in the pull request.

## Reproduction Log

<!-- Append one line per reproduction below this comment. Keep chronological order. -->
