
![logo](https://user-images.githubusercontent.com/57909909/157085532-8f185fa2-d753-4dfe-b834-8e11c2f52e43.jpeg)

# Introduction -:
Santander is a Spanish multinational corporation bank and financial based company which operates in Europe, North and South America, and also Asia. In this Kaggle competition that is conducted by Santander we need to predict whether a customer is dissatisfied with their services early on based on the features provided by the company. This will help them to take proactive steps to improve the customer satisfaction before the customer leaves.
# Business problem: -
- we need to predict whether a customer is satisfied with their services  based on the features provided by the company or not . This will help them to take appropriate action  to improve the customer satisfaction  to stay longer with bank  before the customer leave .
- To build a machine learning algorithm using the training data set and predict the total satisfied and unsatisfied customers in the Santander test data set.

# ML formulation of the business problem:-
Given anonym zed dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.
The task is to predict the probability that each customer in the test set is an unsatisfied customer. Its binary classification problem .
# Data set column analysis : -
- (a)	train.csv - the training set including the target
- (b)	test.csv - the test set without the target
- (c)	sample_submission.csv - a sample submission file in the correct format
- There are 76020 data points 370 columns including the Target variable on train.csv.
- Feature independent names  :- 'var3', 'var15', 'num_var38','imp_op_var40_comer_ult1','imp_op_var40_comer_ult3', 'imp_op_var40_efect_ult1','imp_op_var40_efect_ult3', 'imp_op_var40_ult1'
- Depended Feature:-'The Target variable consists of about 4% of unhappy customers and 96% happy customer. The dataset is highly unbalanced
- Data Source:- https://www.kaggle.com/c/santander-customer-satisfaction/data
# Performance metric:
The metric used here is the area under the ROC .AUC helps in determining whether a model is good at distinguishing between classes The popularity of ROC AUC over standard accuracy measure comes from the fact that in any unbalanced dataset it is easy to achieve 99% accuracy if 99% of the data belongs to one class. Whereas ROC AUC depends on both True Positives as well as False Positives to determine a score thus making the metrics fairly independent of the state of unbalance in the data.
# Contact : -
LinkedIn: https://www.linkedin.com/in/ashish-sarathe-a4b67867/details/skills/

Email: ashish.sarathe@gmail.com 
