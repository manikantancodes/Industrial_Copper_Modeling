# Copper Analysis and Prediction Project

This project aims to analyze copper data and build machine learning models for predicting the selling price and status (Won/Lost) of copper products. The project utilizes Streamlit for creating an interactive web application that allows users to input various features and obtain predictions.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Approach](#approach)
- [Contributing](#contributing)
- [License](#license)
- [Learning Outcomes](#learning-outcomes)

## Introduction

Industrial copper modeling involves analyzing historical copper data to understand trends and patterns. Machine learning models are then trained on this data to predict future outcomes such as selling price and the likelihood of winning a deal (status).

This project implements two main functionalities:

1. **Selling Price Prediction:** Predicts the selling price of copper products based on features such as quantity, thickness, width, country, status, and item type.
2. **Status Prediction:** Predicts the status (Won/Lost) of copper deals based on features similar to the selling price prediction.

## Project Structure

The project consists of the following components:
- `coppermodel.ipynb`: Python script involves data preprocessing, exploratory data analysis (EDA), feature engineering, model building.
- `app.py`: Python script containing the Streamlit web application code.
- `requirements.txt`: File listing the required Python libraries and their versions.
- `__init__.py`: Empty file to mark the directory as a Python package.
- `scaling.pkl`: Pickle file containing the scaler for regression model input.
- `scaling_classify.pkl`: Pickle file containing the scaler for classification model input.
- `ExtraTreeRegressor.pkl`: Pickle file containing the trained regression model.
- `RandomForestClassification.pkl`: Pickle file containing the trained classification model.
- `README.md`: Markdown file containing project documentation.

## Setup

To set up the project locally, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python libraries listed in `requirements.txt`.
 ## Dependencies

The project relies on the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- xgboost
- streamlit

The specific versions of these libraries are listed in the `requirements.txt` file.
 # Copper Analysis and Prediction Project


### Data Understanding
- Identify variable types (continuous, categorical) and distributions.
- Convert invalid 'Material_Reference' values starting with '00000' to null.
- Treat reference columns as categorical variables.

### Data Preprocessing
- Handle missing values using mean/median/mode.
- Treat outliers using IQR or Isolation Forest from sklearn.
- Address skewness with log or boxcox transformations for high skew continuous variables.
- Encode categorical variables with appropriate techniques (one-hot, label, ordinal encoding).

### EDA
- Visualize outliers and skewness with Seaborn's boxplot, distplot, and violinplot.

### Feature Engineering
- Create new features by aggregating or transforming existing ones.
- Drop highly correlated columns using a heatmap.

### Model Building and Evaluation

- Split the dataset into training and testing/validation sets.
- Train and evaluate different classification models such as ExtraTreesClassifier and XGBClassifier.
- Optimize model hyperparameters using techniques such as cross-validation and grid search.

### Model GUI

- Create an interactive page using Streamlit with task input (Regression or Classification).
- Perform feature engineering, scaling, and transformation steps used for training ML model.
- Predict new data in the Streamlit web application.

Run the Streamlit web application using the `streamlit run app.py` command.

## Usage

Once the project is set up and the Streamlit app is running, users can access the web application through their browser. They can input values for quantity, thickness, width, country, status, and item type, and click the "Predict" button to obtain predictions for selling price and status.

### Tips

- Use pickle module to dump and load models such as encoders, scaling models, and ML models.
- Fit and then transform in separate lines and use transform only for unseen data.

## Contributing

Contributions to the project are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
