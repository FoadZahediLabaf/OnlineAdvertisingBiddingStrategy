# OLA Project

## Group Members

| Name | Email |
|---|---|
| Foad Zahedi | foad.zahedi@mail.polimi.it |
| Simone Mauro | simone.mauro@mail.polimi.it |
| Alessandro Sartori | alessandro2.sartori@mail.polimi.it |
| Luigi Cavuto | luigi.cavuto@mail.polimi.it |
| Mengyao Li | Mengyao1.Li@mail.polimi.it |

---

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
- Two smarter exploration variants (UCB1-Tuned and a rescaled-constant UCB), each tested both without and with the budget constraint

### Requirement 2: Multiple Campaigns, Stochastic Environment

Implement and test:

- A stochastic environment with multiple campaigns
- A joint distribution over competing bids
- Combinatorial-UCB with budget constraint
- A conflict graph between non-compatible campaigns

### Requirement 3: Best-of-Both-Worlds, Multiple Campaigns

Implement and test:

- A factorized primal-dual bidding method with a hard budget constraint
- A shared 3000-round, 10-campaign non-stationary market matrix protocol
- Nine sKL/frequency settings with 10 saved realizations per setting
- An exact piecewise-static hindsight oracle and cumulative regret

### Requirement 4: Slightly Non-Stationary Environment

Implement and test:

- Combinatorial-UCB with sliding window
- Combinatorial-UCB with change detector

Compare:
- Sliding window, change detector and primal-dual algorithms

---

## Repository Structure

The repository must stay clean and organized.

    .
    ├── README.md
    ├── CONTRIBUTING.md
    ├── requirements.txt
    ├── pytest.ini
    ├── .gitignore
    ├── ola/
    │   ├── environments/
    │   │   ├── base.py
    │   │   ├── single_campaign.py
    │   │   ├── multi_campaign.py
    │   │   └── non_stationary.py
    │   ├── algorithms/
    │   │   ├── ucb1.py
    │   │   ├── budgeted_ucb.py
    │   │   ├── ucb_variants.py
    │   │   ├── combinatorial_ucb.py
    │   │   ├── primal_dual.py
    │   │   ├── sw_cucb.py
    │   │   └── cd_cucb.py
    │   └── utils/
    │       ├── bidding.py
    │       ├── logging.py
    │       ├── plotting.py
    │       ├── oracle.py
    │       ├── seeding.py
    │       ├── graph.py
    │       └── constellation/
    │           ├── constellation.py
    │           ├── synthetic_dataset.py
    │           └── utils.py
    ├── experiments/
    │   ├── run_req1.py
    │   ├── run_req2.py
    │   ├── run_req3_on_shared_realizations.py
    │   ├── piecewise_static_oracle.py
    │   ├── generate_req4_shared_realizations.py
    │   ├── export_shared_matrix_tables.py
    │   ├── run_req4.py
    │   ├── run_req4_cd.py
    │   ├── run_req4_cd_on_shared_realizations.py
    │   └── run_req4_sw_on_shared_realizations.py
    ├── configs/
    │   ├── req1.yaml
    │   ├── req2.yaml
    │   ├── req3.yaml
    │   └── req4.yaml
    ├── notebooks/
    │   ├── req1_single_campaign.ipynb
    │   ├── req3_primal_dual.ipynb
    │   └── req4_non_stationary.ipynb
    ├── tests/
    │   ├── test_ucb1.py
    │   ├── test_budget.py
    │   ├── test_environments.py
    │   ├── test_req2.py
    │   ├── test_conflict_graph.py
    │   ├── test_constellation.py
    │   └── test_oracle.py
    ├── results/
    │   ├── raw/
    │   ├── processed/
    │   ├── plots/
    │   └── exports/
    └── slides/
        ├── Final Presentation.pptx
        ├── notes.md
        ├── figures/
        └── RequirementsInDeatails/
            ├── Gaussian_Constellation (environment).pptx
            ├── Req1_singleCampaign.pptx
            ├── Req2_Combinatorial_UCB.pptx
            ├── Req3_PrimalDual_current.pptx
            ├── Req4_ComparisonPlots_HiddenPPTX.pptx
            └── Req4_NonStationary.pptx

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
    results/processed/req3_on_req4_shared/req3_factorized_raw.csv
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
| Foad | Project co-lead, repository integration, Requirement 1, code reviewer for the other requirements, and final 3-way comparison: `README.md`, `CONTRIBUTING.md`, `notebooks/req1_single_campaign.ipynb`, `requirements.txt`, `.gitignore`, `ola/__init__.py`, `ola/environments/__init__.py`, `ola/environments/base.py`, `ola/environments/single_campaign.py`, `ola/algorithms/__init__.py`, `ola/algorithms/ucb1.py`, `ola/algorithms/budgeted_ucb.py`, `ola/algorithms/ucb_variants.py`, `experiments/run_req1.py`, `configs/req1.yaml`, `pytest.ini`, `tests/test_ucb1.py`, `tests/test_budget.py`, `tests/test_environments.py`. Foad can review Luigi's, Mengyao's, and Alessandro's code, give consultation, and make code changes only when needed and with the relevant owner's approval. |
| Simone | Project lead, theory, coordination, master slides, and Requirement 4 theory support: `slides/notes.md`, `slides/figures/*`, final presentation material, shared report material, and theory notes for Requirement 4. Simone can also edit Req 3 and Req 4 code when helping Mengyao and Alessandro |
| Luigi | Requirement 2 only: `ola/environments/multi_campaign.py`, `ola/algorithms/combinatorial_ucb.py`, `experiments/run_req2.py`, `configs/req2.yaml` |
| Mengyao | Requirement 3 only: `ola/algorithms/primal_dual.py`, `experiments/run_req3_on_shared_realizations.py`, `experiments/piecewise_static_oracle.py`, `experiments/generate_req4_shared_realizations.py`, `notebooks/req3_primal_dual.ipynb`, and Req3 result/slide artifacts |
| Alessandro | Requirement 4 code only: `ola/environments/non_stationary.py`, `ola/algorithms/sw_cucb.py`, `ola/algorithms/cd_cucb.py`, `experiments/run_req4.py`, `configs/req4.yaml`, `notebooks/req4_non_stationary.ipynb` |
| Team | Final review, dry runs, Q&A preparation, and shared discussion of results. Do not edit code files unless the file owner agrees. |


Result files and test files are not part of personal ownership. They are generated or shared validation artifacts. Do not change them unless the group agrees or the relevant experiment/output needs to be regenerated.

Note: `ola/utils/` is shared by the whole team. Everyone may work on utility files when the change is needed for their requirement, but utility changes must be communicated because they can affect other requirements. If a utility file is only used by one person or one requirement, that person can own that specific change.

Note: Foad is also a project code reviewer. He can review other requirement code, give technical consultation, and suggest changes. If a direct code change is necessary in another owner's files, Foad should get approval from that owner first.

---

## Shared Files That Need Permission

Some shared files have an owner, but changes must still be communicated because they can affect other requirements.

These files affect everyone. Do not edit them without telling the group first:

    requirements.txt
    .gitignore
    ola/environments/base.py
    ola/utils/__init__.py
    ola/utils/bidding.py
    ola/utils/logging.py
    ola/utils/plotting.py
    ola/utils/graph.py
    ola/utils/oracle.py
    ola/utils/seeding.py

---

## Useful Links

Add project links here when available.

    GitHub repository:
    https://github.com/FoadZahediLabaf/ola-project

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
