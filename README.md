# Explaining Errors in Machine Translation with Absolute Gradient Ensembles

Current research on quality estimation of machine translation focuses on the sentence-level
quality of the translations. By using explainability methods, we can use these quality estimations
for word-level error identification. In this work, we present code and results of using different explainability techniques: gradient-based and perturbation-based methods.

## Requirements
Install the packages listed in [requirements.txt](requirements.txt) into a new virtual environment.

```python
conda create -n mt-qe python
conda activate mt-qe
pip install -r requirements.txt
````

## Contents:

Here is a short overview of the repository contents:

- **results**:
    - data:         MLQE-PE train and development sets
    - scripts:      evaluation scripts given by shared task
    - results.ipynb:    our notebook to generate results & ensembles
- **transquest**: adjusted TransQuest-based SHAP and LIME explainers
- **xmover**:
    - xmover: original XMover-Score scripts
    - adjusted XMS-based SHAP and LIME explainers
- **explainability_captum**: Captum-based script for gradient-based & occlusion explainers


## Gradient-based methods + occlusion

The Jupyter notebook **explainability_captum.ipynb** contains our code for all Captum explainers using MonoTransQuest as the quality estimation model. We used Google Colab to run this notebook.

## SHAP and LIME
Our MonoTransQuest (MTQ) and XMover-Score (XMS) based explainers can be found under:

- transquest/transquest_lime.py: MTQ + LIME
- transquest/transquest_shap.py: MTQ + SHAP
- xmover/xmover_explainer_lime.py: LIME + XS
- xmover/xmover_explainer_shap.py: SHAP + XMS

## Results

All of our results are presented in the results directory with individual subfolders for the results of specific language pairs. They also contain the **results.ipynb** Jupyter notebook for evaluating and creating the ensembles.

## References
‘Explainable Quality Estimation’ shared task: https://eval4nlp.github.io/sharedtask.html

## About
This project was carried out as part of the seminar **Meta-Science** at the Technical University of Darmstadt in the summer term 2021.

### Authors:
Chi Viet Vu, chiviet.vu@stud.tu-darmstadt.de \
Jonathan Stieber, jonathan.stieber@stud.tu-darmstadt.de \
Erik Gelbing, erik.gelbing@stud.tu-darmstadt.de \
Melda Eksi, melda.eksi@stud.tu-darmstadt.de
