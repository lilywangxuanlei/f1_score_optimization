## Introduction

Out of all the model I modified, Adaboost was the more challenging to modify yet provides the most room for future improvements. Adaboost output result by combining multiple weak learners into a weighted sum, which represents the final output of the boosted classifier. It typically uses decision trees as weak learners and exponential loss function as loss function. (Wikipedia 2024).  
 
To adapt my method onto Adaboost, logistic classifier was implemented as weaker learners, replacing decision trees, and loss function was customized depending on the model. Within the required timeframe, I was unable to achieve satisfactory results with customized AdaBoost using cross-entropy as the objective function. I, then, compared the performance of the F-1 method with that of the traditional AdaBoost approach. I will discuss more about why Adaboost with cross-entropy as objective function failed and potential modifications to achieve the desired outcome.	

To prevent overfitting, cross-validation was implemented, with 20 as random state. The model includes 40 learners and 100 as max iteration.

## F-1 Method
In F-1 method, the loss function is F-1 method, and the rest is the same as original Adaboost model. 
 
When testing a balanced dataset, the average F-1 score is 0.9139, with the score of each fold being 0.9126, 0.8981, 0.9262, 0.9221, and 0.9106.

When testing an imbalanced dataset, the average F-1 score is 0.8478, with the score of each fold being 0.8589, 0.8350, 0.8660, 0.8440, and 0.8349.

[code](Adaboost_f1.py)

## Original Method
Adaboost function from scikit-learn library was implemented.
 
When testing a balanced dataset, the average F-1 score is 0.9519, with the score of each fold being 0.9532, 0.9474, 0.9467, 0.9532, and 0.9588.

When testing an imbalanced dataset, the average F-1 score is 0.9198, with the score of each fold being 0.9231, 0.9174, 0.9132, 0.9302, and 0.9151.

[code](Adaboost_og.py)

## Cross-entropy 
When trying to customize the model with cross-entropy as objective function, I struggled with updating sample weight, alpha effectively and processing logits or predictions incorrectly. 
 
When testing a balanced dataset, the average F-1 score is 0.0895, with the score of each fold being 0.0980, 0.1095, 0.0667, 0.0864, and 0.0871.

When testing an imbalanced dataset, the average F-1 score is 0.0575, with the score of each fold being 0.0458, 0.0896, 0.0348, 0.0617, and 0.0554.

[code](Adaboost_Cross_entropy.py)

## Observation and Interpretation

Since the Cross-Entropy method is not fully developed, the discussion is regarding the comparison between F-1 method and the original method. According to the data, F-1 method performs worse than the original Method by 3.99% on balanced data and 7.83% on imbalanced data. This appears to be a satisfactory result for the F-1 method, as it closely aligns with the performance of the Adaboost algorithm which uses DecisionTrees as classifiers and an exponential loss function. 
