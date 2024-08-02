# Project Title: Predictive Analytics for Employee Retention: A Machine Learning Approach to Analyzing Employee Churn
### Author: Jarrod Sims - [Linkedin](www.linkedin.com/in/jarrod-sims-a7467a94)
Full project report: https://www.overleaf.com/read/zmqswytxqxtb#075cee

## Project Overview
Employee turnover can adversely impact an organization’s financial outlook and performance, as well as crucial elements for company success such as reputa- tion, culture, and morale. Companies accrue costs associated with recruiting and training replacements, and costs such as paying out unused PTO. While most companies generally try to avoid employee churn due to the reasons mentioned above, there are situatiWith the implementation of HRIS, organizations have been gathering more quantity of data that has not been used in its full potential yet. Thus, interest in performing research on using predictive  analytics for employee churn has been growing in recent years.ons where turnover can be beneficial for a company’s bottom-line. In situations where high-performing employees are replaced with less experienced or lower-performing individuals, the remain- ing employees must shoulder the burden of the departing employee. Conversely, when a poor performer is replaced with a high-performing individual, company productivity increases. In the latter scenario, retaining a poor-performing em- ployee for too long can be detrimental to the company.With the implementation of HRIS, organizations have been gathering more quantity of data that has not been used in its full potential yet. Thus, interest in performing research on using predictive  analytics for employee churn has been growing in recent years.


## Project Phases
1. **Data Collection**
    - The dataset for this project, sourced from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), encompasses human resources data for 1,470 current and former IBM employees. It includes 36 features covering various demographic and performance-related aspects of employee satisfaction and attrition. 
2. **Data Cleaning & Transformation**
    - Data cleaning and preprocessing was performed using Python as well as the Pandas library.
3. **Exploratory Data Analysis (EDA)**
    - This process includes the following steps:
        1. Importing necessary data and libraries
        2. Describing data using summary statistics
        3. Defining correlations between variables and identifying patterns in data using visuals

4. **Model Selection & Training**
5. **Analysis of Results**

## Exploratory Analysis

Figure 2 illustrates that younger individuals are more prone to leaving the company compared to their older counterparts. The median age for those who experienced positive attrition was 32 years, whereas it was 36 years for those who did not leave. For individuals who left the company, 50% were between the ages
of 28 and 39, while 50% of those who stayed were between 31 and 43 years old. This clear disparity underscores the greater likelihood of younger employees departing from the company.


![age_attrition_plot](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/plot_age_attrition.png)

An employee’s monthly income showed a correlation with attrition. Figure 6 below depicts the income distribution differences between employees who left the company and those who remained. Notably, 50% of the employees who left had a monthly salary ranging from $2,373 to $5,916, with a median salary of $3,202. In contrast, 50% of the employees who stayed earned between $3,211 and $8,834 per month, with a median salary of $5,204.

![salary_attrition_plot](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/salary_attrition_plot.png)

Departments within the company were analyzed to identify those with higher attrition rates. The code below creates a DataFrame that combines the ’Attrition’ and ’Department’ fields and calculates the percentage of attrition for each department. Figure 3 illustrates that, although the three major departments had relatively similar attrition rates, sales and human resources experienced notably higher rates compared to research and development, with attrition rates of 20.6% and 19.0%, respectively. 


![department_attrition_plot](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/department_attrition_plot.png)

## Model Selection and Training
**Training and Testing**

The dataset was split into a training set that consist of 80% and a test set of 20%. Each model was trained on the same training and test dataset. 

![splitter_code](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/splitter_code.png)

This project evaluated four models to classify employee attrition, decision trees, KNN, XGBoost, and random forest. Below is a comparison of their performance:

**Classifier Model Development**

![classifier_performance](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/classifier_performance_eval.png)

Extreme Gradient Boosting (XGBoost) is a technique that utilizes a similar approach to decision trees but instead of a single tree, builds multiple trees and Title Suppressed Due to Excessive Length 13 iterates on each tree by correcting errors made by earlier trees through gradient descent boosting. The XGBClassifier class from the XGBoost library was employed to implement a XGBoost model on the attrition dataset.

![XGBoost_code](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/xgboost_classifier_code.png)

XGBoost classifier feature importance plot.

![XGBoost_feature_importance_plot](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/xgboost_feature_importance_plot.png)

**Prescriptive Model Development**

Proximal Policy Optimization (PPO) represents a category of reinforcement learning algorithms. These algorithms are notable for their effectiveness across a diverse array of reinforcement learning scenarios. PPO stands out due to its straightforward implementation, requiring fewer hyperparameters for tuning compared to other prevalent methods like Q-learning. This simplicity makes PPO an attractive option for various applications in the field. See PPO prescriptive model implementation at:[ppo_prescriptiverl.py](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/ppo_prescriptiverl.py). See PPO model output below:

![PPO_model_output](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/ppo_model_output.png)

## **Project Setup and Installation:**
1. Create a virtual environment
   ```
   python3 -m venv ds-venv
   ```

2. Activate the environment
     - On Windows:
      ```
      ds-venv\Scripts\Activate
      ```
     - On macOS and Linux
      ```
      source env/bin/activate
      ```


### Ensure that you have the following packages installed:
- Python 3
- Jupyter Notebook
- Python libraries listed in [reqirements.txt](https://github.com/simsjarrod/MachineLearning_Employee-Churn/blob/master/requirements.txt)

### Clone GitHub Repository:
To clone a GitHub repository, follow these steps:

1. **Copy the Repository URL**
    : https://github.com/simsjarrod/MachineLearning_Employee-Churn

2. **Open a Terminal or Command Prompt**

    Open a terminal (Linux/Mac) or Command Prompt (Windows).

3. **Run the `git clone` Command**

In your terminal or command prompt, navigate to the directory where you want to clone the repository. Then, use the `git clone` command followed by the URL you copied:
```sh
git clone https://github.com/simsjarrod/MachineLearning_Employee-Churn
```

## 
**Author:**
Jarrod Sims

**Date Updated:**
August 2nd, 2024

**Acknowledgments:**
The project utilizes ChatGPT, an AI language model developed by OpenAI, for assistance in writing the README and providing guidance on software engineering practices.
