## On the Reproducibility of "FairCal: Fairness Calibration for Face Verification"

Code for On the Reproducibility of "FairCal: Fairness Calibration for Face Verification". Based on the code for the paper FairCal: Fairness Calibration for Face Verification (https://github.com/tiagosalvador/faircal)

## Requirements
TODO Ryan write this

To install requirements:

```setup
conda install --file requirements.txt
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
