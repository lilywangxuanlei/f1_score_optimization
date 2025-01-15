As mentioned in the Dataset section, traditional F-1 score does not correctly reflect performance of the model when the data is highly skewed, since the result will have a high recall and reasonable precision. This flaw can be improved by applying macro-F-1 score, that does not take label imbalance into account. This improvement could be extended to F-1 loss function in F-1 method to improve the model performance on highly skewed data. 

Adaboost with cross-entropy as objective function has big room in improvements. The problems I run into were regarding the update of logits and probability, weighted update, and how to make sure the loss converge, even with the help with optimizer. 

While I was doing research for Cross-entropy, I discovered the function of Binary-Cross-Entropy from Scikit-learn library, which produce really accurate result for models when implemented correctly. However, I did not have enough time to investigate this function. I think trying to improve the performance of F-1 model by comparing the result to BCE functions could yield meaningful result. 

I think there is potential to improve F-1 score so it performs better with large data set as well. Also Apply the idea of Macro F-1 to F-1 method, or test the result using Macro F-1 Score.
