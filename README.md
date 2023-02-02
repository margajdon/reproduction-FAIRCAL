## On the Reproducibility of "FairCal: Fairness Calibration for Face Verification"

Code for On the Reproducibility of "FairCal: Fairness Calibration for Face Verification". Based on the code for the paper FairCal: Fairness Calibration for Face Verification (https://github.com/tiagosalvador/faircal)

## Data

Two data sources are used:
- Balanced Faces in the Wild (BFW) - https://github.com/visionjo/facerec-bias-bfw
- Racial Faces in the Wild (RFW) - http://whdeng.cn/RFW/testing.html

One must fill out a form to obtain the BFW dataset and send an email request to obtain the RFW dataset.

Once the data has been obtained, it should only require the unzipping of the data and place the necessary folders as described in the filestructure.txt.

## Requirements

This repository was tested on Linux and OS. Windows users beware.

Running this repository requires working conda base environment: https://www.anaconda.com

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

## Notes about important files

**run_all.py**: Contains an all-in-one function to create the data, run the experiments and save the outputs

**generate_embeddings.py**: Contains the FacenetEmbeddingGenerator, WebfaceEmbeddingGenerator and ArcfaceEmbeddingGenerator classes used to generate the embeddings from the image dataset using the Facenet, Facenet-Webface and Arcface model respectively

**approaches.py**: The main class in this file is the ApproachManager which is used to run the different approaches (Baseline, FairCal, FSN, Agenda, FairCal-GMM, Oracle). The ApproachManager class inherits from AgendaApproach and FtcApproach methods that are specific to the Agenda and FTC approaches respectively.

**csv_creator.py**: File used to generate csv templates and manage dataframes.



