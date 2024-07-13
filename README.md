# Heart Disease Classification

## Description
This is an example project that demonstrates a full code base to train a model to classify the presence of heart disease. There is a standard module for handling the different dataframes, and a Flask app has been created for hosting and serving the model. While full feature engineering, hyperparameter tuning, etc. have been performed due to time constraints, the overaching approach is representative of the approach and methodology necessary for the successful development of a production machine learning model. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [ROI KPIs](#roi-kpis)
- [Success Criteria](#success-criteria)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [References](#references)
- [Appendix](#appendix)
- [Credits](#credits)

## Installation
To use, deploy, and/or retrain this machine learning model, install the dependencies listed in the `requirements.txt` file.

## Usage
1. For local API usage, deploy the Flask app by:
   1. Navigating to the root direction `..\Heart_Disease_Classifcation`
   1. Executing the script: `python -m app.inference_script` (this will launch a flask app, likely with an address of `http://127.0.0.1:8000`)
1. Generate inference by following the steps outline in `notebooks/Endpoint_Example.ipynb`

For an offline (non-API) example, see `notebooks/Create_Results.ipynb`.

## Data
Data was sourced from kaggle (`mexwell/heart-disease-dataset`). If you would like to source it net-new, please execute `scripts/1_fetch_data.py` from the root directory (e.g., `python -m scripts.1_fetch_data`)

## Model Training
For this basic example, I created a Feed Forward Neural Network using PyTorch. Specifically, it has 2 hidden layers with each layer having a dimension of 100. See `config.py` for model specifics.

## Evaluation
Two files have been created to demonstrate how evaluation could be performed: `scripts/4_evaluate_model.py` and `notebooks/Endpoint_Example.ipynb`

In the notebook, you'll see an example creation of a confusion matrix to evaluate both True and False predictions.

## ROI KPIs
This particular project does not necessarily have any ROI KPIs since it is an example, but here, we would outline specific KPIs & targets and how we plan to measure them moving forward (e.g., reduction in YoY Diagnosis of Heart Disease for this model)

## Success Criteria
We will consider our project successful if we achieve the following:
1. Develop a fully functional machine learning pipeline from data preprocessing to model training, evaluation, and orchestrated deployment.
2. Achieve a baseline performance metric that exceeds a predefined threshold.
3. Iterate on the pipeline by augmenting with additional data, performing feature engineering, and incorporating feedback from end users.
4. Improve the performance metric by a significant margin compared to the baseline.

## Contributing
In order to contribute to this project, please follow GitHub best practices: submit enhancements / issues / feature requests via GitHub Issues; make code contributions on new branches (pull requests to main will be reviewed prior to merging)

## Roadmap
Likely valuable to outline the future plans and goals for the project. Include any upcoming features, improvements, or bug fixes.

## Frequently Asked Questions (FAQ)
Address common questions or concerns that users may have about the project.

## References
List any relevant articles, papers, or documentation that users can refer to for further information.

## Appendix
Include any additional information or resources that may be helpful for users.

## Credits
Erik Howard (erikdhoward@gmail.com)


