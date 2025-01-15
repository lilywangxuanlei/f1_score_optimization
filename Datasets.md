The data set implemented on models is Student_performance_data, for illustration purposes. 

In dataset Student_performance_data, Balanced data refers using Binary classification of GradeClass with below 3 (includes 3) as 1 and the rest as 0. In this case, the ratio of 0:1 is 1211: 1181. And imbalanced data classify GradeClass below 2 (includes 2) as 1 and the rest as 0. In this case, the ratio of 0:1 is 1625: 767. The imbalanced data set is not highly skewed. 

In dataset HW_5_data, the data is imbalanced when the output is binary (Take 1): the ratio of 0:1 is 16836: 4645

Testing on HW-5 data is very challenging. Not only because it is unbalanced but also it is very large. Some of the empirical data I got does not support my claim, like neural network model. I think this could be because I did not tune hyper-parameter correctly. Or it could be my method requires more adjustment to improve its prediction ability. Unfortunately, I was not able to find the solution within the time frame. I have listed what I tried below and what could be improved.

Features of the data is normalized using StandardScaler

From my observation. When, for example, 1 takes up 90% of the data, the model tends to classify every data as 1 and still yields a high f-1 score. This is because when the model predicts every data as 1, we will have a high recall, because there are only few FN and a reasonable Precision because there are only few 0 in the data and therefore, FP will be low as well. 

This flaw and possible solutions will be continually discussed in the Limitations and Future Improvement sections. 
