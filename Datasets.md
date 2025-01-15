The data set implemented on models is Student_performance_data, for illustration purposes. 

For example, in data Student_performance_data, Balanced data refers using Binary classification of GradeClass with below 3 (includes 3) as 1 and the rest as 0. In this case, the ratio of 0:1 is 1211: 1181. And imbalanced data classify GradeClass below 2 (includes 2) as 1 and the rest as 0. In this case, the ratio of 0:1 is 1625: 767. The imbalanced data set is not highly skewed. 

Features of the data is normalized using StandardScaler

From my observation. When, for example, 1 takes up 90% of the data, the model tends to classify every data as 1 and still yields a high f-1 score. This is because when the model predicts every data as 1, we will have a high recall, because there are only few FN and a reasonable Precision because there are only few 0 in the data and therefore, FP will be low as well. 

This flaw and possible solutions will be continually discussed in the Limitations and Future Improvement sections. 
