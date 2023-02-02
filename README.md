## On the Reproducibility of "FairCal: Fairness Calibration for Face Verification"

Code for On the Reproducibility of "FairCal: Fairness Calibration for Face Verification". Based on the code for the paper FairCal: Fairness Calibration for Face Verification (https://github.com/tiagosalvador/faircal)

## Requirements

This repository was tested on Linux and OS. Windows users beware.

Running this repository requires working conda base environment:
```
https://www.anaconda.com
```

To create the conda environment to run the repo, first create the fact2023 conda environment:
```
conda env create -f fact_environment.yml
```
Activate the fact2023 environment: 
```
conda activate fact2023
```
Install the pip packages:
```
pip install mxnet
pip install facenet-pytorch
pip install pycave
```
Recommended for OS users:

```
conda install -c conda-forge nomkl
```

## Preparing data


To run the experiments, the image embeddings, pairs and cosine similarities need to be generated. The full pipeline, including running all experiments, can be run using 
```
python run_all.py
```

Due to the licenses of the datasets, the embeddings, pairs and cosine similarities for both the RFW and BFW datasets cannot be shared

## Evaluating Methods

To run all experiments, run the following command
```
python main_fairness_analysis.py
```

To run with specific datasets, features, approaches or calibration method, run
```
python main_fairness_analysis.py --datasets [datasets] --features [features] --approaches [approaches] --calibration_methods [calibration_methods] 
```

## Figures and Tables

The notebooks that were used to generate the tables and figures used can be found in the notebooks folder. 
