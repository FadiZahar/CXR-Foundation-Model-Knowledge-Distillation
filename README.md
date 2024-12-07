# **Foundation Models for Chest Radiography:** Knowledge Distillation and Impacts on Performance and Bias Propagation

![Alt text](<assets/CXRFMKD - Diagram.png>)

Welcome to the GitHub repository of my postgraduate research thesis for the *MSc in Artificial Intelligence* at Imperial College London. This project was carried out under the supervision of Dr. Ben Glocker in the [Biomedical Image Analysis (BioMedIA) Lab](https://biomedia.doc.ic.ac.uk/).


## Table of Contents

- [Project Overview](#project-overview)
  - [Motivation](#motivation)
  - [Key Contributions](#key-contributions)
- [Datasets](#datasets)
- [Setup](#setup)
  - [Python Environment Setup](#python-environment-setup)
  - [System Requirements](#system-requirements)
  - [CXR-FM: Generate Feature Embeddings](#cxr-fm-generate-feature-embeddings)
  - [CheXpert & MIMIC: Generate Study Data](#chexpert--mimic-generate-study-data)
- [Repository Structure](#repository-structure)
- [Get in Touch](#get-in-touch)
- [References](#references)


## Project Overview

### Motivation
This project was initiated through an examination of Google’s Chest X-Ray (CXR) Foundation Model (FM) [**[1]**](#references), hereafter referred to as **CXR-FM**. The investigation focused on leveraging Knowledge Distillation (KD) to mitigate the biases identified by [**[2]**](#references) in this FM and to enhance its overall performance.

Historically, CXR-FM, like many foundation models, was constrained by a lack of publicly accessible weights, providing only vector embedding outputs from input images. This limited access impedes efforts towards bias mitigation, as full access to the model's architecture is essential for comprehensive adjustments. 

To address these limitations, KD techniques were employed to extract and transfer insights from the CXR-FM 'teacher' to a newly developed 'student' model, named **CXR-FMKD**, which is based on the DenseNet169 architecture. This approach aimed to overcome the transparency issues inherent in proprietary models and improve their safe applicability and performance in clinical settings.

>Recently, aligning with the themes of this research, the [weights for CXR-FM were made public](https://research.google/blog/helping-everyone-build-ai-for-healthcare-applications-with-open-foundation-models/), setting an important precedent in a competitive AI landscape where sharing insights from proprietary models is not typically incentivised. This shift further highlights the ongoing necessity for open access to medical foundation models and transparency in their development to drive innovation and improve safety across healthcare applications.

> **Note:** For a Motivation overview, please refer to the accompanying [Motivation PowerPoint PDF](<assets/CXRFMKD - Motivation.pdf>)

### Key Contributions

This research was centered on developing the **CXR-FMKD** model using Knowledge Distillation (KD) techniques to distill insights from the proprietary **CXR-FM** and transfer them to a DenseNet169-based architecture.

Key contributions include:

$\textbf{\color{crimson}{1. KD Exploration:}}$

- Various loss functions—$\color{darkblue}{MSE}$, $\color{darkblue}{MAE}$, $\color{darkblue}{HuberLoss}$, $\color{darkblue}{Cosine\ Similarity}$—and their combinations were investigated in this KD process. The best-performing CXR-FMKD models were obtained using a combination of *MSE* and *Cosine Similarity*.

$\textbf{\color{crimson}{2. Performance\ Analysis:}}$

- The performance of the CXR-FMKD model was evaluated using the CheXpert and MIMIC datasets, which both encompass the same 14 disease classes in a multilabel classification setting. Performance metrics included $\color{darkcyan}{AUC–PR}$, $\color{darkcyan}{AUC–ROC}$, $\color{darkcyan}{Maximum\ Youden's\ J\ Statistic}$, and $\color{darkcyan}{Youden's\ J\ Statistic\ at\ 20\%\ FPR\ (False\ Positive\ Rate)}$.

- The CXR-FMKD consistently outperformed both the original CXR-FM and a benchmark DenseNet169 model trained from scratch.

$\textbf{\color{crimson}{3. Generalisability\ Analysis\ (Inference):}}$

- Models initially trained on CheXpert were subsequently tested on MIMIC to assess generalisability, leveraging the shared 14 disease classes. Various testing strategies were employed, including $\color{saddlebrown}{Direct\ Transfer}$ without adaptation, $\color{saddlebrown}{Linear\ Probing}$ by fine-tuning only the final classifier layer, and $\color{saddlebrown}{Full\ Fine–Tuning}$ across all layers. These tests confirmed the robustness of CXR-FMKD and further demonstrated its enhanced performance compared to CXR-FM.

$\textbf{\color{crimson}{4. Bias\ Analysis:}}$

- $\color{green}{PCA}$ and $\color{green}{t–SNE}$ dimensionality reduction techniques were used to analyse variations in subgroup marginal distributions in the penultimate layer embeddings for biases related to age and sex. To quantify these variations, a novel $\color{olivedrab}{bias\ score}$ was developed using outputs from $\color{green}{Kolmogorov–Smirnov}$ $\color{green}{statistical\ tests}$, which were categorised by p-value significance levels and assigned predefined scores. This bias analysis highlighted a notable reduction in bias in CXR-FMKD compared to CXR-FM.

$\textbf{\color{crimson}{5. Performance\ vs.\ Bias\ Analysis:}}$

- A general trend was observed where $\color{purple}{increased\ performance}$ **correlated with** $\color{purple}{reduced\ bias}$, suggesting that the models were utilising more relevant features in the chest X-rays for disease classification, rather than inappropriate reliance on features like race or sex.

These findings underscore the potential of KD not only to effectively mitigate bias but also to enhance performance for specific data tasks. Importantly, this work further highlights the crucial need for transparency in foundation models, where restricted access to model weights through API-dependent platforms often obscures vital details necessary for deeper understanding and bias mitigation.

> **Note:** For an overview of the experimental results, please refer to the accompanying [Results PowerPoint PDF](<assets/CXRFMKD - Results.pdf>)

## Datasets

This project utilises two key Chest X-Ray (CXR) datasets:

- **CheXpert Dataset**:
    - The CheXpert CXR dataset can be downloaded from https://stanfordmlgroup.github.io/competitions/chexpert/ 
    - The corresponding demographic information is available at https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf

- **MIMIC Dataset**:
    - The MIMIC CXR dataset can be downloaded from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ 
    - The corresponding demographic information is available at https://physionet.org/content/mimiciv/1.0/


## Setup

### Python Environment Setup
To ensure the proper execution of the code, setting up a dedicated Python environment is recommended. Below are the instructions for a Virtualenv setup:

1. **Create and Activate a Python 3 Virtual Environment:**
    ```bash
    python3 -m venv <path_to_envs>/fmkd_venv
    source <path_to_envs>/fmkd_venv/bin/activate
    ```

2. **Install Required Python Libraries:**

    Ensure your virtual environment is activated, then install the libraries specified in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

### System Requirements
*Model training and testing* require high-end GPU workstations. For our experiments, two main GPU partitions were used:

1. ***gpus24 Partition:***

- *GPUs*: Workstation-grade RTX 3090 / 4090 cards.
- *GPU Memory*: Each GPU has 24 GiB of VRAM.
- *System Specs*: Accompanied by 62 GiB RAM and 12 CPU cores.
- *Usage*: Used for various tasks in the project, predominantly when lesser memory was sufficient.

2. ***gpus48 Partition:***

- *GPUs*: Server-grade Ada A6000 / L40 cards.
- *GPU Memory*: Each GPU has 48 GiB of VRAM.
- *System Specs*: Accompanied by 125 GiB RAM and 16 CPU cores.
- *Usage*: Used for running the Knowledge Distillation (KD) codes and other demanding high-memory tasks.

*Model output analysis and plotting* can be performed on standard laptop computers. Specifically, for our experiments, a MacBook Pro with the following specifications was used:

- *Processor*: Apple M2 Pro with a 12-core CPU and a 19-core GPU
- *Memory*: 16GB of unified RAM


### CXR-FM: Generate Feature Embeddings

This work analyses CXR-FM, the CXR foundation model by Google Health. Detailed instructions for using this model to produce feature embeddings for the CXR (CheXpert and MIMIC) datasets can be found in the [original GitHub repository](https://github.com/Google-Health/imaging-research/tree/master/cxr-foundation).


### CheXpert & MIMIC: Generate Study Data

To effectively replicate the study data used in this research, please refer to a previous [BioMedIA repository](https://github.com/biomedia-mira/chexploration?tab=readme-ov-file) which provides detailed instructions on generating and resampling the datasets. 

This process is essential for preparing the `TRAIN_RECORDS_CSV`, `VAL_RECORDS_CSV`, and `TEST_RECORDS_CSV` files, which are utilised in the code and referenced in the configuration files `config/config_chexpert.py` for CheXpert and `config/config_mimic.py` for MIMIC.

#### Instructions for Setup and Execution:

1. **Download and Preparation**:

    Refer to the [BioMedIA repository](https://github.com/biomedia-mira/chexploration?tab=readme-ov-file) for precise instructions on how to download the CheXpert and MIMIC datasets and where to add the corresponding csv files for use in the section below.

2. **Generating Study Data**:

    Execute the `chexpert.sample.ipynb` and `mimic.sample.ipynb` notebooks as outlined in the repository to generate the necessary study data. These notebooks will guide you through the process of sampling from the datasets to create representative training, validation, and test sets.

3. **Performing Test-Set Resampling**:

    Follow the steps in the `chexpert.resample.ipynb` notebook to perform test-set resampling. This notebook details the method for balancing the datasets based on demographic and clinical characteristics.

#### Configuration Files:

Ensure the paths to the generated CSV files (`TRAIN_RECORDS_CSV`, `VAL_RECORDS_CSV`, `TEST_RECORDS_CSV`) are correctly set in `config/config_chexpert.py` for CheXpert and `config/config_mimic.py` for MIMIC.


### Repository Structure

This section outlines the repository's organisational structure and shows how different parts of the project correlate to the [key contributions](#key-contributions) made in this research:

```
cxr-fmkd/
├── analysis/                               # Analysis of the model's performance and biases
│   ├── bias_analysis/                      # Bias Analysis (Contribution 4)
│   ├── bias_vs_performance_analysis/       # Performance vs. Bias Analysis (Contribution 5)
│   └── performance_analysis/               # Performance Analysis (Contributions 2 & 3)
├── assets/                                 # Supplementary materials including images and PDF files
├── config/                                 # Configuration files for datasets and models
├── data_modules/                           # Data preprocessing modules
├── inference/                              # Inference methods for model evaluation
│   ├── full_finetuning/                    # Full Fine-Tuning (Contribution 3)
│   ├── linear_probing/                     # Linear Probing (Contribution 3)
│   └── zero_shot/                          # Direct Transfer (Contribution 3)
├── models/                                 # Model architectures and training scripts
│   ├── knowledge_distillation/             # KD Exploration (Contribution 1)
│   └── ...                                 # Models training for disease classification (Contributions 1 & 2)
└── utils/                                  # Utility scripts
```

Refering back to the [Key Contributions](#key-contributions):

1. *Contribution 1*: $\textbf{\color{crimson}{KD Exploration}}$
2. *Contribution 2*: $\textbf{\color{crimson}{Performance\ Analysis}}$
3. *Contribution 3*: $\textbf{\color{crimson}{Generalisability\ Analysis\ (Inference)}}$
4. *Contribution 4*: $\textbf{\color{crimson}{Bias\ Analysis}}$
5. *Contribution 5*: $\textbf{\color{crimson}{Performance\ vs.\ Bias\ Analysis}}$


## Get in Touch

For more information or to inquire about the research, please reach out directly at fadi.zahar23@imperial.ac.uk.

## References

> [1] Sellergren AB, Chen C, Nabulsi Z, et al (2022)
Simplified Transfer Learning for Chest
Radiography Models Using Less Data. Radiology
305:454–465

> [2] Glocker B, Jones C, Roschewitz M, Winzeck S
(2023) Risk of Bias in Chest Radiography Deep
Learning Foundation Models. Radiol Artif Intell
5:230060


