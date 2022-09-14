# Fake news classifier  

This project focuses on fake news detection problem and provides simple pipeline for that task.  

## Basic project setup
To use this project you need to do the following:
1. Clone this repository with `git clone`
2. Create virtual environment with Python 3.10 inside project directory.
3. Activate newly created virtual environment
4. Install basic dependencies* with `pip install -r requirements.txt`
5. While in main directory setup** project with `pip install .` 
6. Create `data` folder in main project directory. In this folder place both `bodies.csv` and `stances.csv`

Now the project is fully functional and can be used to classify fake news.

Note 1: both `requirements.txt` and `requirements_dev.txt` are required to utilize pytest, mypy and flake8

Note 2: flag `-e` can be used to install project in editable mode

## Other information
It is advised to first read through `headline-body_relation_analysis.ipynb` to fully understand thought process behind this project.

Both `xgb_train.ipynb` and `full_test_xqb_pipeline.ipynb` can be used to quickly create and test new XGBoost models.

Configuration of most training and testing utilities is located in `config.toml` file.

## Plans for future
- Refactor classification pipeline to allow usage of different kinds of models
- Add more sophisticated training loop for XGBoost that would allow k-fold Cross Validation and hyper-parameter optimization
- Move project to stand-alone scripts
- Full test coverage

Note: GitHub Actions isn't working properly
