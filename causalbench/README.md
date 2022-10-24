# BeCause

The project compares Causal Discovery and Bayesian Structure Learning for cross-sectional data on a fair basis.

## Folder structure
* data: data for experiments. Each dataset has its own folder. Some data set have multiple versions. 
* experiment: contains script(s) for benchmarking algorithm and jupyter notebook(s) to visualize results. 
* result: experiment results are saved here. Ideally, this should be where local MLflow database is saved in the later development.
* src: contains R and Python modules. Some modules are copied directly from other (unmaintained) packages or repositories. Disclaimers are given in such case. R scripts are saved in the subfolder R. Modules that are not in the R-folder are all written in Python.

## Using Causal-Discovery-Toolbox
The Python package *causal-discovery-toolbox (cdt)* requires installing R and some R-packages that are not so intuitive to install. To make the package (partially) usable:
1. Install R
1. (optional) install R studio
1. Run the R-script `/src/R/cdt_install_rpackages.R`
1. Copy the file path to your Rscript.exe
1. In your Python script, set path to your Rscript.exe after importing cdt, e.g.
``` 
import cdt
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.0.5/bin/Rscript.exe'
 ```

# Example experiment
The script `/experiment/example_experiment.py` contains example code to compare algorithm performance. The code is not bug-free and not efficient. To run the script
1. Open the script. Find the line `cdt.SETTINGS.rpath =` and change the path to your Rscript.exe path.
1. **option 1** Run the code directly in your IDE (e.g. VS code or PyCharm)
**option 2** Run the code in your terminal (I only know for Windows).
   2.1. Go to BeCause folder
   2.2. Type `python -m experiment.example_experiment.py`

Running the script with `python experiment\example_experiment.py` will give `ModuleNotFoundError` because the script is in a subfolder, which tries to call `varsortability` in another subfolder.