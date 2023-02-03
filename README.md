# On the Reproducibility of "FairCal: Fairness Calibration for Face Verification"

Code for On the Reproducibility of "FairCal: Fairness Calibration for Face Verification". Based on the code for the paper FairCal: Fairness Calibration for Face Verification (https://github.com/tiagosalvador/faircal)

## Data

Two data sources are used:
- Balanced Faces in the Wild (BFW) - https://github.com/visionjo/facerec-bias-bfw
- Racial Faces in the Wild (RFW) - http://whdeng.cn/RFW/testing.html

One must fill out a form to obtain the BFW dataset and send an email request to obtain the RFW dataset.

Once the data has been obtained, it should only require the unzipping of the data and place the necessary folders as described in `filestructure.txt`.

## Pretrained models

The pretrained models should be downloaded automatically (just once) and all necessary code to run them work out of the box. However, documentation can be found at:
- Facenet models: https://github.com/timesler/facenet-pytorch
- Arcface model: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface

## Requirements

This repository was tested on Linux and MacOS. Windows users beware.

Running this repository requires working conda base environment: https://www.anaconda.com

To create the conda environment to run the repo, first create the mlrc2022 conda environment:
```
conda env create -f mlrc_environment.yml
```
Activate the mlrc2022 environment: 
```
conda activate mlrc2022
```
Install the pip packages:
```
pip install mxnet
pip install facenet-pytorch
pip install pycave
```
Recommended for non-Intel users:

```
conda install -c conda-forge nomkl
```

## Preparing data


To run the experiments, the image embeddings, pairs and cosine similarities need to be generated. The full pipeline, including running all experiments, can be run using 
```
python run_all.py
```

Due to the dataset licensing, the embeddings, pairs and cosine similarities for both the RFW and BFW datasets cannot be shared.

## Evaluating Methods

To run all experiments, run the following command
```
python fairness_analyzer.py
```

To run with specific datasets, features, approaches or calibration method, run
```
python fairness_analyzer.py --datasets [datasets] --features [features] --approaches [approaches] --calibration_methods [calibration_methods] 
```

## Figures and Tables

Figures and tables were generated via notebooks, and can be executed after running all experiments.

## Notes about important files

**run_all.py**: Contains an all-in-one function to create the data, run the experiments and save the outputs


**fairness_analyzer.py**: This is the main file where the fairness experiments occur after the generation of the embeddings. The two main classes in the file are RfwFairnessAnalyzer and BfwFairnessAnalyzer, which contain all the attributes and methods specific to each dataset. The common methods are inherited from the FairnessAnalyzer class.

**generate_embeddings.py**: Contains the FacenetEmbeddingGenerator, WebfaceEmbeddingGenerator and ArcfaceEmbeddingGenerator classes used to generate the embeddings from the image dataset using the Facenet, Facenet-Webface and Arcface model respectively

**approaches.py**: The main class in this file is the ApproachManager which is used to run the different approaches (Baseline, FairCal, FSN, Agenda, FairCal-GMM, Oracle). The ApproachManager class inherits from AgendaApproach and FtcApproach methods that are specific to the Agenda and FTC approaches respectively.

**cosine_similarity_cals.py**: Thie file contains the functions that load the template containing the image pairs and their metadata, maps the embeddings that were previously generated and derives cosine similarities.

**csv_creator.py**: File used to generate csv templates and manage dataframes.

**calibration_methods.py**: Contains the calibration classes. In the original paper and the reproduction paper, the main focus was on the Beta calibration.

**dependencies**: Folder that contains the dependencies for the Arcface model. Files are from: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface





