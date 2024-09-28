# Accounting Fraud Detection using Contextual Language Learning

This repository contains the Python code accompanying the paper: [Accounting Fraud Detection using Contextual Language Learning](https://www.sciencedirect.com/science/article/pii/S1467089524000150#s0015).

The code is organized into three main sections:

1. **Data Preparation**  
2. **Results**  
3. **Supplementary Analysis**

### Data Preparation

The data preparation process is divided into two parts. First, we construct the yearly text data, then we combine it with financial data to produce the final ensemble dataset. For further details, please refer to Section 3 of the paper.

#### Text Data Construction

The process of constructing the text data is explained in the folder: ["Text_Data_Preparation"](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Data_Preparation/Text_Data_Preparation). We collect Management Disclosure and Analysis (MDA) reports from 1994 to 2013, capturing the raw text data from these reports. The detailed steps, from URL collection to text preparation, are described in Section 3.1 of the paper. 

The processed MDA reports for each year can be found in the folders: `Text_Data_Preparation/MDA_{year}`. This approach is inspired by the work of [Berns et al. (2022)](https://doi.org/10.1111/fire.12280).

The final yearly text aggregation is handled by the scripts in the ["Combining_All_Years"](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Data_Preparation/Text_Data_Preparation/Combining_All_Years) folder.

#### Financial and Ensemble Data Preparation

To create the ensemble dataset, the text data is combined with financial data, which is prepared based on the features described in [Bao et al. (2020)](https://doi.org/10.1111/1475-679X.12292). The scripts for assembling the final ensemble data are located in the ["Ensemble_Data_Preparation"](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Data_Preparation/Ensemble_Data_Preparation) folder.

### Results

The scripts for generating results are located in the ["Results"](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Results) folder. These include:

- **BERT Predictions**: Scripts in the [BERT_Predictions](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Results/BERT_Predictions) folder.
- **LDA Predictions**: Scripts in the [LDA_Predictions](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Results/LDA_Predictions) folder.
- **RusBoost**: Scripts for this model are in the [RusBoost](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Results/Parameter_Tuning/RusBoost) folder.

Each model’s parameter tuning scripts are located in the ["Parameter_Tuning"](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Results/Parameter_Tuning) folder. For more information, refer to Section 5 of the paper.

### Supplementary Analysis

Additional analyses, including tackling class imbalance, robustness checks, exclusion of serial frauds, model ensembling, and retraining BERT models on an ensemble data subset, are covered in the ["Supplementary_analysis"](https://github.com/IndranilUvA/IJAIS-Fraud/blob/bb642599cafde8391a0a78ce87eaa3f87ec9b28c/Supplementary_analysis) folder. Detailed discussions are available in Section 7 of the paper.
