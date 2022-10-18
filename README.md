# dFC-fUS: Dynamic functional connectivity analysis for functional ultrasound data.

dFC-fUS is a Python package for analyzing dynamic functional connectivity in fUS data.

## Usage

### Data processing pipeline
The package contains several stand-alone scripts, comprising the fUS data processing pipeline:

* ```data_import.py``` (Importing data from CUBE server)
* ```run_app.py``` (Web-app for data visualization and selecting brain regions in brain warps)
* ```sICA.py``` (Spatial ICA pipeline)
* ```pre_processing.py``` (Pre-processing pipeline)
* ```main_processing.py``` (Deconvolution and HMM inference)
* ```main_processing_separate_groups.py``` (HMM inference on separate groups of mice)

All scripts share settings in the ```configurations.ini``` file.
To run the entire pipeline, you should insert your settings in the ```configurations.ini``` file and 
subsequently run ```tutorial.py```. Note: you need full access to all data for running the entire pipeline.

### Simulations
For creating simulations, modify and run the ```simulations.py``` script.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the software. Run the following commands within
a virtual environment:

```bash
git clone https://github.com/rubenwijnands999/dFC-fUS
cd dFC-fUS
pip install .
```
Note: cloning might take a while due to a large example data set included.

## Compatibility
Software was tested on MacOS Monterey 12.2.1 with Python 3.7.

## MSc Thesis
[TU Delft Repository](https://repository.tudelft.nl/islandora/object/uuid%3Ae4692392-9010-4875-8392-6801513277c5)
