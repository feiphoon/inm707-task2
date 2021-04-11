# File tree

This is the file tree up to a certain depth.

The most important files for marking are listed here. Any discrepancies will only be due to more experiment results being logged or processed.

The `maze` folder contains Task 2 (with code completed as Task 1). All results are logged in the results folder, under their respective datetime runs, and any associated figures, images and text files (individual model detailed figures CSV and text environment print to `std.out`) are with them. Each datetime run folder also has a `results_summary.csv` of all experiments run at the time.

```bash
.
├── README.md
├── requirements.txt
├── src
│   ├── example.py
│   └── maze
│       ├── e_greedy_policy.py
│       ├── maze.py
│       ├── q_maze.py
│       ├── qlearning.py
│       ├── qlearning_analysis.ipynb
│       ├── qlearning_exp_functions.py
│       ├── qlearning_exp_runner.ipynb
│       ├── results
│       │   └── 20210411-041749
│       ├── README.md
│       └── utils.py
├── tasks.py
└── tests
    ├── __init__.py
    └── test_maze.py
```
