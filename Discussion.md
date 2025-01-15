Supported by the data, model with F-1 method generally has better performance than model with other traditional method by at least 2% and even greater on imbalanced dataset. 

In imbalanced datasets, the majority class dominates, and standard loss functions (e.g., Cross-Entropy Loss) can lead to models that focus on predicting the majority class to minimize overall loss. F-1 method avoid this problem why minimize FP and FN as well. It penalizes models that perform poorly on either precision or recall, ensuring a more balanced tradeoff between false positives (FP) and false negatives (FN).

Another important note is that under some conditions, models with cross-entropy perform better than F-1 method, occurs when large dataset was tested, or /and when the model is not tuned properly. For example, F-1 method performs worse with a small number of epochs than the traditional method, or an improper learning rate.

As observed in Neural Network model: F-1 method is a lot more stable, obtains higher F-1 score than traditional method when the data number is not large. However, this is not practical, since we will have way larger data set than HW5. F-1 method is not stable when the number is large.
