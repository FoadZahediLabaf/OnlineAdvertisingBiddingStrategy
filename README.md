# OLA Project

## Project Overview

Online Learning Applications project.

The goal of this project is to design, implement, test, and compare online learning algorithms for bidding in advertising campaigns under budget constraints.

We are building a simulation framework for repeated advertising auctions.

At each round, the advertiser chooses bids for one or more campaigns. If a bid is high enough, the advertiser wins the auction, receives a reward, and spends part of the budget.

The project focuses on:

- Online learning
- Multi-armed bandits
- Budget constraints
- Multiple advertising campaigns
- Conflict constraints between campaigns
- Stochastic environments
- Non-stationary environments
- Empirical comparison of algorithms

---

## Project Requirements

### Requirement 1: Single Campaign, Stochastic Environment

Implement and test:

- UCB1 ignoring the budget constraint
- UCB1 extended with a budget constraint

### Requirement 2: Multiple Campaigns, Stochastic Environment

Implement and test:

- A stochastic environment with multiple campaigns
- A joint distribution over competing bids
- Combinatorial-UCB with budget constraint
- A conflict graph between non-compatible campaigns

### Requirement 3: Best-of-Both-Worlds, Multiple Campaigns

Implement and test:

- A highly non-stationary or adversarial environment
- A primal-dual bidding method with budget constraint
- A full-feedback setting where the highest competing bid is observed

### Requirement 4: Slightly Non-Stationary Environment

Implement and compare:

- Combinatorial-UCB with sliding window
- Combinatorial-UCB with change detector
- Primal-dual method

---

## Repository Structure

The repository must stay clean and organized.

    .
    ├── README.md
    ├── CONTRIBUTING.md
    ├── requirements.txt
    ├── .gitignore
    ├── ola/
    │   ├── environments/
    │   │   ├── base.py
    │   │   ├── single_campaign.py
    │   │   ├── multi_campaign.py
    │   │   ├── adversarial.py
    │   │   └── non_stationary.py
    │   ├── algorithms/
    │   │   ├── ucb1.py
    │   │   ├── budgeted_ucb.py
    │   │   ├── combinatorial_ucb.py
    │   │   ├── primal_dual.py
    │   │   ├── sw_cucb.py
    │   │   └── cd_cucb.py
    │   └── utils/
    │       ├── bidding.py
    │       ├── logging.py
    │       └── graph.py
    ├── experiments/
    │   ├── run_req1.py
    │   ├── run_req2.py
    │   ├── run_req3.py
    │   └── run_req4.py
    └── configs/
        ├── req1.yaml
        ├── req2.yaml
        ├── req3.yaml
        └── req4.yaml

---

## Repository Rules

- Do not work directly on `main`.
- Do not delete files.
- Do not rename files.
- Do not move files.
- Do not create random new files.
- Do not change the repository structure without group approval.
- Do not edit files unrelated to your task.
- Do not overwrite other people’s work.
- Do not change the CSV result format after the interface freeze without group approval.
- Do not change CSV column names.
- Ask the group before making structural changes.

---

## Where Code Should Go

### Environments

Use this folder for environment classes and simulation logic:

    ola/environments/

### Algorithms

Use this folder for learning algorithms:

    ola/algorithms/

### Utilities

Use this folder for helper functions, plotting utilities, logging, graph tools, and bidding utilities:

    ola/utils/

### Experiments

Use this folder for scripts that run experiments:

    experiments/

### Configurations

Use this folder for experiment configuration files:

    configs/

---

## Files Not to Create

Do not create files like:

    test.py
    final.py
    copy.py
    new_version.py
    my_code.py
    temporary.py

Use the existing structure instead.

---

## Installation and Environment Setup

Each person must set up their own Python environment before running the project.

This avoids problems caused by different Python versions or missing libraries.

### 1. Clone the Repository

Use this command to download the repository:

    git clone <repository-url>

Then enter the project folder:

    cd <repository-folder>

Replace `<repository-url>` with the actual GitHub repository link.

Replace `<repository-folder>` with the folder created after cloning.

### 2. Create a Virtual Environment

On macOS or Linux:

    python3 -m venv .venv

On Windows:

    python -m venv .venv

This creates a local environment called `.venv`.

### 3. Activate the Virtual Environment

On macOS or Linux:

    source .venv/bin/activate

On Windows PowerShell:

    .venv\Scripts\Activate.ps1

On Windows Command Prompt:

    .venv\Scripts\activate.bat

After activation, the terminal should show something like this:

    (.venv)

This means the virtual environment is active.

### 4. Install Requirements

After activating the virtual environment, install the required libraries:

    pip install -r requirements.txt

### 5. Updating Requirements

Do not install random libraries without telling the group.

If a new library is necessary:

- Tell the group first.
- Explain why the library is needed.
- Install it only after the group agrees.
- Then update `requirements.txt`.

To update `requirements.txt`:

    pip freeze > requirements.txt

Do not update `requirements.txt` with unnecessary libraries.

---

## How to Run Experiments

Before running any experiment, activate the virtual environment.

On macOS or Linux:

    source .venv/bin/activate

On Windows:

    .venv\Scripts\activate

Then run the experiment script.

Example:

    python experiments/run_req1.py

Each experiment should:

- Run without manual changes.
- Use reproducible parameters.
- Save results to a CSV file.
- Save plots or allow plots to be regenerated from saved CSV files.

---

## Reproducibility Rules

Experiments must be reproducible.

This means another group member should be able to run the same script and obtain the same type of output.

Each experiment should clearly define:

- Number of rounds `T`
- Number of campaigns `N`, if applicable
- Budget
- Bid set
- Campaign values
- Environment parameters
- Random seed, if used
- Algorithm parameters

Do not hide important parameters inside unclear code.

Use configuration files when possible.

Recommended folder for configuration files:

    configs/

Example:

    configs/req1.yaml

---

## Output and Results Format

Every experiment must save a CSV file with the same required column names.

Use this exact required order:

    algorithm,round,action,bids,wins,reward,cost,budget_left,cumulative_reward,cumulative_cost,benchmark_reward,instant_regret,cumulative_regret

Do not rename, remove, or reorder the required columns.

Extra columns are allowed only after the required columns.

Do not change the CSV result format after the interface freeze without group approval.

The official result format is defined consistently in `README.md`, `CONTRIBUTING.md`, and `ola/utils/logging.py`.

If everyone uses the same format, we can generate comparison plots easily.

Allowed extra columns include:

    seed
    environment_name
    config_name
    highest_competing_bid
    done

Extra columns must come after the required columns.

---

## Required Columns

| Column | Meaning |
|---|---|
| `algorithm` | Name of the algorithm that produced the row |
| `round` | Current round number |
| `action` | Selected action or selected arm |
| `bids` | Bid or vector of bids |
| `wins` | Won auctions |
| `reward` | Reward obtained in the round |
| `cost` | Cost paid in the round |
| `budget_left` | Remaining budget after the round |
| `cumulative_reward` | Total reward up to the current round |
| `cumulative_cost` | Total cost up to the current round |
| `benchmark_reward` | Benchmark reward for comparison |
| `instant_regret` | Regret in the current round |
| `cumulative_regret` | Total regret up to the current round |

---

## Action, Bids, and Wins Format

For Requirement 1:

- `action` is the selected bid.
- `bids` is the selected bid as a scalar.
- `wins` is `0` or `1`.

Example:

    action,bids,wins
    0.4,0.4,1

For Requirements 2, 3, and 4:

- `action` may identify the selected feasible combination or policy.
- `bids` should contain one bid per campaign.
- `wins` should contain one win indicator per campaign.

Example:

    action,bids,wins
    feasible_set_0,"[0.1, 0.4, 0.2]","[1, 0, 1]"

---

## Missing Benchmark Values

If benchmark or regret is not implemented yet, keep `benchmark_reward`, `instant_regret`, and `cumulative_regret` as empty values.

These columns must still exist:

    benchmark_reward
    instant_regret
    cumulative_regret

Do not delete them.

---

## Results Folder

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

## Plotting and Reporting Rules

Plots must be generated from saved CSV files.

Do not use screenshots as final results.

Every plot should have:

- A clear title
- An x-axis label
- A y-axis label
- A legend when comparing multiple algorithms

Useful plots include:

- Cumulative reward over time
- Cumulative regret over time
- Budget left over time
- Cumulative cost over time
- Algorithm comparison plots

Final plots should be reproducible from saved result files.

---

## Git Reminder

Do not work directly on `main`.

Work on your own branch.

Only merge into `main` when:

- The code is finished.
- The code runs without errors.
- The experiment works.
- The result CSV has the required columns.
- The plots can be regenerated from saved data.
- The group agrees.

For the full Git workflow, read:

    CONTRIBUTING.md

---

## File Ownership and Access Rules

Each person should work only on the files assigned to their task.

Do not edit files outside your assigned area unless the owner agrees first.

Do not delete, rename, move, or create files without group approval.

If a file is listed in the table below and does not exist yet, the assigned owner is allowed to create it.

For any other new file, ask the group before creating it.

Use the file names below unless the group agrees on a different name.

---

## Allowed Files by Owner

| Owner | Files / folders they can edit |
|---|---|
| Foad | `README.md`, `CONTRIBUTING.md`, `.gitignore`, `requirements.txt`, `ola/environments/base.py`, `ola/environments/single_campaign.py`, `ola/algorithms/ucb1.py`, `ola/algorithms/budgeted_ucb.py`, `ola/utils/bidding.py`, `ola/utils/logging.py`, `experiments/run_req1.py`, `configs/req1.yaml`, `results/raw/req1_*.csv`, `results/processed/req4_3way_comparison*.csv` |
| Luigi | `ola/environments/multi_campaign.py`, `ola/algorithms/combinatorial_ucb.py`, `ola/utils/graph.py`, `experiments/run_req2.py`, `configs/req2.yaml`, `results/raw/req2_*.csv` |
| Luigi + Simone | `ola/environments/adversarial.py`, `ola/algorithms/primal_dual.py`, `experiments/run_req3.py`, `configs/req3.yaml`, `results/raw/req3_*.csv` |
| Mengyao + Simone | `ola/environments/non_stationary.py`, `ola/algorithms/sw_cucb.py`, `ola/algorithms/cd_cucb.py`, `experiments/run_req4.py`, `configs/req4.yaml`, `results/raw/req4_*.csv` |
| Simone + Team | Final slides only, presentation notes, and shared final-report material. Do not edit code files for this unless the file owner agrees. |

Note: `ola/utils/graph.py` is mainly used by Requirement 2, but changes must be communicated because Requirement 3 and Requirement 4 may also use it.

---

## Shared Files That Need Permission

Some shared files have an owner, but changes must still be communicated because they can affect other requirements.

These files affect everyone. Do not edit them without telling the group first:

    requirements.txt
    .gitignore
    ola/environments/base.py
    ola/utils/bidding.py
    ola/utils/logging.py
    ola/utils/graph.py

---

## Useful Links

Add project links here when available.

    GitHub repository:
    <repository-link>

    Master slides:
    <slides-link>

    Shared notes:
    <notes-link>

    Colab notebooks:
    <colab-links>

---

## Final Checklist

Before saying your part is finished:

- [ ] The code runs without errors.
- [ ] The correct files were edited.
- [ ] No random files were created.
- [ ] No files were deleted.
- [ ] The repository structure was not changed.
- [ ] The output CSV uses the required columns.
- [ ] The results are saved.
- [ ] The plots can be regenerated from saved data.
- [ ] The code is pushed to GitHub.
- [ ] The group understands what changed.