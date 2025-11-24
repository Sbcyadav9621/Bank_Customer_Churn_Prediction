# Bank_Customer_Churn_Prediction
The Objective of this Project is to predict/classify customers who are inactive(Churn) or Active(Not Churn) based on historical data by implementing Machine Learning Models

## Why do we need machine learning model for Churn prediction why not others
churn prediction is fundamentally "Classification problem", "With uncertainty", "With complex patterns". Machine learning models are designed exactly for this, Learning from past data to predict future outcomes.
Machine Learning Models have following capabilities:
  1. Handles large & complex data
  2. ML can process hundres of features simultaneously.
        - Transaction behaviour
        - App usage
        - Demographics
        - Historical patterns
  3. Learns patterns automatically.
  4. Predicts future behaviour
        ML is predictive, not just descriptive.
         - Predicts who whill churn before it happens
         - Enables proactive retention strategies
  5. Adapts over time
         Models can be retrained as cutomer behaviour changes, keeping accuracy high.

We use a machine learning model for churn prediction because customer behavior is complex and cannot be accurately captured using simple rule-based logic or manual thresholds. Traiditional methods fail to detect hidden patterns and interactions between multiple factors such as usage, payment history, and engagement trends. A machine leanring model automatically learns these patterns from historical data, adapts over time, and provides probabilistic predictions, allowing the business to proactively identify customers at risk of churn and take retention actions.

# Approach to Solve this Project
  1. Data Extraction
  2. Data Audit or Data Check
  3. Data Cleaning
  4. Exploratory Data Analysis
  5. Data Preprocessing ( Feture Selection + Feature Engineering)
  6. Model implementation
  7. Modle Evaluation
  8. Hyper tuning (Based on model performance)
  9. Conclusion.

# 1. Data Extraction
  I get this data through Kaggle website. I load this data into "Google Colab".
# 2. Data Audit or Data Check
  Perform basic data audit on imported data. checking shape, size, columns, feture data types, any mis matches, 
# 3. Data Cleaning
  Checek before doing. 
  - Checking null values. In this data set there is no null values
  - Checking duplicated records. There is no duplicated records.
# 4. Exploratory Data Analysis
  This is the crucial step in entire Machine learning model. We must do exploratory data analysis. EDA simply means explore data, and find what is happend in that data.
  - divide categorical features into one side and numerical features into another side.
  - check correlation on numerical features and visualize correlation
  - perform Univariate analysis
  - perform Bivariate analysis
  - Check Outliers using boxplot
  - Outlier capping using IQR method.(do not perfrom outlier capping on target columns)
# 5. Data Preprocessing ( Feture Selection + Feature Engineering)
  - Label Encoding using LabelEncoder. It simply converts categorical feature values into numerical ones.
# 6. Model Implemantation
  - Implementing DecisionTreeClassifier model using default parameters
  - ### Training accuracy- 100%  & Testing Accuracy - 79%
  - Decision Tree model is over fitted.
    ### Possible techniques to tackle overfitting issue.
    1. Pruning the Decision Tree
       Limit tree growth by setting parameters such as:
         * max_depth
         * min_samples_split
         * min_samples_leaf
    2. Hyperparameter tuning using GridSearchCV
       Optimize tree parameters to find the best balance between bias and varaince
    3. Cross-validation
       Use K-Fold Cross Validation to ensure the model generalizes well to unseen data
    4. Feature selection / Dimensionality reduction
       Remove irrelevant or highly correlated features to reduce noise and complexity.
    5. Ensemble methods
        Use models like "RandomForest", "GradientBoosting" which reduce overfitting by averaging multiple trees.
    ### I use Cross-Validation method first & Hyper parameter tuning using GridSearch CV then Ensemble methods
      The average cross-validation score is 79%
    ### Hyper parameter tuning using GridSearchCV
      After applying Hyper parameter tuning using GridSearchCV i got 86% Accuracy in training ad 86% Accuracy in Testing also
      The model is not overfitted and not underfitted. It is perfectly generalized model.

    # RandomForest Model 
    The RandomForest Classifier model with default parameters got 100% accuracy in training and 87% accuracy in testing.
    #### Training accuracy - 100%  & Testing accuracy - 87%

    ### Random Forest Model with hyper parameters
    After applying Hyper parameter tuning i got train accuracy 90% and 86% in test
      #### Training accuracy - 90% & Testing Accuracy - 86%
    
    # XGBoost Classifier model
      The XGBoost classifier model is overfitted.
       #### Training accuracy - 96%  &  Testing accuracy - 86%

    # XGBoost Classifier with Hyper parameters
      The XGBoost classifer with hyper parameters got 88% in traning and 87% in testing.

    # Accuracy Table
    ------------------------------------------------------------------
    Model namel                    | Train Accuracy  | Test Accuracy
    ------------------------------------------------------------------
    Decision Tree Classifier       |     100%        |     78%

    Decision Tree Hyper parameter  |     86%         |     86%

    Random Forest Classifier       |     100%        |     87%

    Random Forest Classifier Hyper |
    Parameters                     |      90%        |     86%

    XGBoost Classifier             |      96%        |     86%

    XGBoost Classifier             |      88%        |     87%
    ------------------------------------------------------------------

    # Conclusion:
      Out of all 6 models, XGBoost Classifier with best hyper parameters got good accuracy in both training and testing set.  Which indicated Generalized model. No underfitting and overfitting issues.
    
    
    
    
    

