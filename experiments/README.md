# Example of how to run experiment in command-line interface

## Preparation

- Python file for experiment

  In this example, the path to the python file is `CausalBench\experiments\castle\experiment_castle_json_args.py`. This file can be used as a template for setting up experiments. It can run experiments with all available datasets and all integrated algorithms from gCastle.

- Input JSON file

  You can find a JSON file in the path: `CausalBench\experiments\example_input.json`. It contains 3 parts: datasets, algorithms, and the name of the file where the results of the experiment will be stored.

## Run the experiment

First, go to the directory of the experiment python file. In this example, it's:

```shell
cd CausalBench/experiments/castle
```

Then use this command to run the experiment: `python3 -m experiment_name.py -src relative/path/to/input.json`. In the case of this example:

```shell
python3 -m experiment_castle_json_args -src ../example_input.json
```

## The result file

The result file will be created (if not already exists) during the execution of the experiment. In the folder where you find the experiment python file, a `result` folder will be created (if not already exists). All experiments running with that experiment python file will be stored in that `result` folder. In the case of this example, the CSV file which stores the result data of the experiment locates in `CausalBench\experiments\castle\result\out_put_file_name.csv`

```
├── castle
│   ├── experiment_castle_json_args.py
│   └── result
│       └── output_file_name.csv
└── example_input.json
```

## How to customize the JSON file?

- Datasets, algorithms, and output file name in the JSON file can be customized.
- All supported datasets and algorithms can be found in a template JSON file `params_template.json` in `CausalBench\experiments`
- Different algorithms and different datasets support different arguments as input. Please use the `params_template.json` as a reference.
