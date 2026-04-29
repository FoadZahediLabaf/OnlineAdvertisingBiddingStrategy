# Contributing Guide

## Purpose

This file explains how every group member should work on the project.

The goal is to keep the repository clean, organized, and working.

Everyone must follow the same Git workflow so that we do not lose code, overwrite each other’s work, or break the project.

---

## Main Collaboration Rules

- Do not work directly on `main`.
- Work on your own branch.
- Pull the latest version before starting work.
- Commit only files related to your task.
- Do not delete files.
- Do not rename files.
- Do not move files.
- Do not create random files.
- Do not change the repository structure without group approval.
- Do not change the CSV result format after the interface freeze without group approval.
- Do not change CSV column names.
- Do not overwrite other people’s work.
- Ask the group if you are unsure.

---

## Branch Workflow

The `main` branch must contain only stable and finalized code.

Every person must create a separate branch for their own task.

The normal workflow is:

1. Start from `main`.
2. Pull the latest version.
3. Create your own branch.
4. Work on your branch.
5. Commit your changes.
6. Push your branch.
7. Test everything.
8. Merge only when the work is finished and approved.

---

## Branch Naming Convention

Use clear branch names.

Good examples:

    feature/req1-ucb
    feature/req1-budgeted-ucb
    feature/req2-combinatorial-ucb
    feature/req3-primal-dual
    feature/req4-sliding-window
    feature/req4-change-detector
    fix/plot-error
    fix/budget-bug
    docs/update-readme

Bad examples:

    test
    new
    branch1
    final
    mybranch
    random

The branch name should explain what you are working on.

---

## Before Starting Work

Go inside the project folder:

    cd <repository-folder>

Switch to `main`:

    git checkout main

Download the latest version:

    git pull origin main

Create your own branch:

    git checkout -b feature/your-task-name

Example:

    git checkout -b feature/req1-ucb

After creating the branch, check that you are not on `main`:

    git branch

The active branch has a `*` next to it.

Good:

    * feature/req1-ucb
      main

Bad:

    * main

If you are on `main`, stop and switch to your branch before editing files.

---

## Making Changes Safely

Before editing a file, make sure it is related to your task.

Do not change files just because they look wrong or unclear.

If you think a file needs to be changed but it is not part of your task, ask the group first.

Before editing any file, check the "Allowed Files by Owner" table in `README.md`.

Ownership summary:

- Foad owns repository integration, Requirement 1, shared utilities, tests, and the final 3-way comparison output.
- Simone owns project coordination, theory support, the master slides, and final presentation/report material.
- Luigi owns Requirement 2 files only.
- Mengyao owns Requirement 3 files only.
- Alessandro owns Requirement 4 code files only.

If the file is not assigned to you, ask the owner before editing it.

Before making changes, check the current status:

    git status

This shows which files are modified.

If you see files that you did not change intentionally, do not commit them.

---

## File Rules

Do not delete files.

Do not rename files.

Do not move files.

Do not create random files.

Do not change the folder structure.

Do not change files unrelated to your task.

Do not change the CSV result format after the interface freeze without group approval.

Do not change CSV column names.

Do not overwrite other people’s work.

If you need a new file that is not listed in the "Allowed Files by Owner" table in `README.md`, ask the group first.

---

## Files You Must Not Commit

Do not commit temporary files, cache files, or personal files.

Examples:

    __pycache__/
    .ipynb_checkpoints/
    .DS_Store
    test.py
    copy.py
    final.py
    new_version.py
    temporary.py

If you see these files in `git status`, remove them before committing.

---

## Code Rules

Write clean and readable code.

Use clear variable names.

Avoid very long functions.

Add comments only when the logic is not obvious.

Do not add useless comments.

Before adding a new helper method or function, check the existing files in `ola/utils/` carefully.

Use existing utility methods and functions whenever possible.

Add a new utility only if the needed functionality does not already exist.

If you add a new utility function, keep it general, reusable, and clearly named.

Do not duplicate logic that already exists in `ola/utils/`.

Bad variable names:

    x
    y
    z
    temp
    stuff

Better variable names:

    selected_bid
    remaining_budget
    cumulative_reward
    number_of_rounds
    campaign_values

Bad comment:

    # add one

Better comment:

    # Update the number of pulls after selecting the arm.

---

## Result Format Rules

Every experiment must save results using the standard CSV format defined in `README.md` and `ola/utils/logging.py`.

Do not rename columns.

Do not remove columns.

Do not change the meaning of columns.

Do not reorder the required columns.

Do not change the CSV result format after the interface freeze without group approval.

The official result format is defined consistently in `README.md`, `CONTRIBUTING.md`, and `ola/utils/logging.py`.

The required columns are:

    algorithm,round,action,bids,wins,reward,cost,budget_left,cumulative_reward,cumulative_cost,benchmark_reward,instant_regret,cumulative_regret

The required columns must appear first and in this exact order.

Extra columns are allowed only after the required columns.

Allowed extra columns include:

    seed
    environment_name
    config_name
    highest_competing_bid
    done

Extra columns must come after the required columns.

### Action, Bids, and Wins Format

For Requirement 1:

- `action` is the selected bid.
- `bids` is the selected bid as a scalar.
- `wins` is `0` or `1`.

For Requirements 2, 3, and 4:

- `action` may identify the selected feasible combination or policy.
- `bids` should contain one bid per campaign.
- `wins` should contain one win indicator per campaign.

### Missing Benchmark Values

If benchmark or regret is not implemented yet, keep `benchmark_reward`, `instant_regret`, and `cumulative_regret` as empty values.

These columns must still exist:

    benchmark_reward
    instant_regret
    cumulative_regret

Do not delete them.

## Results Folder Rules

Raw experiment CSV files go in:

    results/raw/

Processed or aggregated result files go in:

    results/processed/

Generated plots go in:

    results/plots/

Final slide-ready figures go in:

    slides/figures/

Raw CSV file naming convention:

    results/raw/req<requirement_number>_<algorithm_name>.csv

Examples:

    results/raw/req1_ucb1.csv
    results/raw/req1_budgeted_ucb.csv
    results/raw/req2_combinatorial_ucb.csv
    results/raw/req3_primal_dual.csv
    results/raw/req4_sw_cucb.csv
    results/raw/req4_cd_cucb.csv

Plot naming convention:

    results/plots/req<requirement_number>_<metric>.png

Examples:

    results/plots/req1_cumulative_reward.png
    results/plots/req1_budget_left.png
    results/plots/req1_cumulative_cost.png
    results/plots/req1_selected_bid.png

Rules:

- Do not overwrite another person’s result file without asking.
- Do not save final results only as screenshots.
- Do not change result file names randomly.
- Final plots must be generated from saved CSV files.

---

## Commit Rules

Before committing, always check what changed:

    git status

Read the list carefully.

Only commit files related to your task.

Do not commit random files.

Do not commit broken code.

Do not commit temporary result files.

Commit only final or comparison result files that the group needs for plots and presentation.

---

## How to Add Files

Add one specific file:

    git add path/to/file.py

Example:

    git add ola/algorithms/ucb1.py

Add multiple specific files:

    git add ola/algorithms/ucb1.py experiments/run_req1.py configs/req1.yaml

Add all changed files:

    git add .

Use `git add .` only after checking `git status`.

---

## How to Commit

Use a short and clear commit message.

Example:

    git commit -m "Implement UCB1 algorithm"

Good commit messages:

    git commit -m "Implement budgeted UCB"
    git commit -m "Add single campaign environment"
    git commit -m "Fix budget update bug"
    git commit -m "Add Req 1 experiment script"
    git commit -m "Generate Req 1 result plots"

Bad commit messages:

    git commit -m "update"
    git commit -m "fix"
    git commit -m "changes"
    git commit -m "final"
    git commit -m "stuff"

A good commit message should explain what changed.

---

## Push Rules

After committing, push your branch to GitHub:

    git push origin your-branch-name

Example:

    git push origin feature/req1-ucb

If this is the first time pushing a new branch, Git may ask you to set the upstream branch.

Use:

    git push -u origin your-branch-name

Example:

    git push -u origin feature/req1-ucb

Do not push directly to `main` unless the work is finalized and the group agrees.

---

## Merge Rules

Before merging into `main`, your work must be finished and tested.

Do not merge unfinished code.

Do not merge code that gives errors.

Do not merge code that changes unrelated files.

Do not merge code that changes the repository structure without approval.

Do not merge code that changes the CSV result format after the interface freeze without group approval.

Only merge when:

- The code runs without errors.
- The experiment works.
- The result CSV has the correct columns.
- The plots can be regenerated from saved data.
- The branch is pushed to GitHub.
- The group agrees.

---

## Correct Merge Process

First, update `main`:

    git checkout main
    git pull origin main

Then go back to your branch:

    git checkout your-branch-name

Example:

    git checkout feature/req1-ucb

Merge the latest `main` into your branch:

    git merge main

Test your code again:

    python experiments/run_req1.py

If everything works, push your updated branch:

    git push origin your-branch-name

Then merge your branch into `main`:

    git checkout main
    git pull origin main
    git merge your-branch-name
    git push origin main

Only do this when the work is finalized and approved.

---

## Pull Before You Start Every Time

Every time you start working, update your branch with the latest version of `main`.

Use:

    git checkout main
    git pull origin main
    git checkout your-branch-name
    git merge main

This reduces the chance of conflicts.

---

## Handling Merge Conflicts

A merge conflict happens when two people changed the same part of the same file.

If Git shows a conflict, do not panic.

Do not randomly delete code.

Do not accept changes without reading them.

Open the conflicted file and look for something like this:

    <<<<<<< HEAD
    code from main
    =======
    code from your branch
    >>>>>>> your-branch-name

You must carefully decide which code should stay.

After fixing the file, save it.

Then run:

    git add path/to/conflicted-file.py
    git commit -m "Resolve merge conflict"

If you do not understand the conflict, ask the group before doing anything.

---

## Dangerous Git Commands

Some Git commands can delete work.

Do not use these commands unless the group agrees and you understand what they do.

Do not use:

    git reset --hard

Do not use:

    git push --force

Do not use:

    git clean -fd

These commands can remove local work or overwrite other people’s work.

If you think you need one of these commands, ask the group first.

---

## Before Saying Your Task Is Finished

Before saying your task is finished, check everything carefully.

Your task is finished only when:

- The code runs without errors.
- The experiment works.
- The result CSV has the correct columns.
- The plots can be regenerated from saved CSV files.
- You changed only files related to your task.
- You did not create random files.
- You did not delete files.
- You did not change the repository structure.
- You committed your changes.
- You pushed your branch to GitHub.
- You told the group what you changed.

---

## Final Checklist Before Merging

Before merging into `main`, check:

- [ ] I am on my own branch.
- [ ] My branch is updated with the latest `main`.
- [ ] The code runs without errors.
- [ ] The experiment script runs.
- [ ] The result CSV has the required columns.
- [ ] The plots can be regenerated.
- [ ] I did not change unrelated files.
- [ ] I did not delete files.
- [ ] I did not create random files.
- [ ] I did not change the repository structure.
- [ ] I pushed my branch.
- [ ] The group agrees that this can be merged.

---

## Final Reminder

Never work directly on `main`.

Always work on your own branch.

Always pull before starting.

Always check `git status` before committing.

Always test before merging.

Ask the group if you are unsure.