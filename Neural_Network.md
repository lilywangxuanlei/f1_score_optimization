## Introduction

  In neural network, each artificial neuron receives signals from connected neurons, then processes them and sends a signal to other connected neurons. The "signal" is a real number, and the output of each neuron is computed activation function. The strength of the signal at each connection is determined by a weight, which adjusts during the learning process (Wikipedia 2024). I defined the NN model with a residual connection between layers to help mitigate vanishing gradients and improve training. The network structure consists of one input layer, two hidden layers, each with 20 neurons and ReLU activations, and one output layer for binary classification. Predictions are made by applying a sigmoid activation to the logits and thresholding at 0.5. The loss function is customized to align with the approach of F-1 method or traditional method. Both models use Adam Optimizer to minimizing loss, with learning rate of 0.001 and are trained for 50 Epochs. Model used cross-validation to prevent over-fitting, and within each fold, it is trained using mini batches (size 32). 
## F-1 Method

  In this model, the loss function is approximated F1-score as demonstrated earlier. Since Neural Network (NN) is a well-developed deep learning model, the primary customization was its loss function.  

Student_performance_data::

  When tested on a balanced dataset, the average F-1 score is 0.9416, with the score of each fold being 0.9389, 0.9324, 0.9417, 0.9573, and 0.9374.

  When tested on an imbalanced dataset, the average F-1 score is 0.8890, with the score of each fold being 0.9013, 0.8805, 0.9072, 0.8815, and 0.8746.

Hw_5_data:
When batch size =64, neurons in hidden layer = 32, epoch = 200, learning rate =0.001, F-1 method yield an average F-1 score of 0.6893 with each fold equals to 0.7119, 0.6799, 0.6831, 0.6922, and 0.6796. 

[code](Neural_Network_f1.py)

## Cross-entropy

  In this model, loss function is BCEWithLogitsLoss () from Torch library.

Student_performance_data:

  When testing a balanced dataset, the average F-1 score is 0.9261, with the score of each fold being 0.9320, 0.9203, 0.9357, 0.9201, and 0.9224.

  When testing an imbalanced dataset, the average F-1 score is 0.8709, with the score of each fold being 0.8805 0.8601, 0.8630, 0.8966, and 0.8545.
  
  HW5_data:
  Cross-Entropy yields an average of F-1 score of 0.5146 with each fold equals to: 0.4112, 0.4862, 0.5556, 0.4418, 0.6780. F-1 method outperforms Cross-entropy by 38.34%

  [code](Neural_network_cross_entropy_hw5.py)

## Observation and Interpretation

  Supported by the data, f1 method performed 1.67% better than cross-entropy method on balanced data and 2.03% better on imbalanced data. Neural Network is well-developed deep learning model, which I believe explains why the F-1 method only performs slightly better than the cross-entropy method on both balanced data and imbalanced data. Neural networks can effectively model complex, non-linear relationships between data points by using multiple layers of interconnected nodes. This robustness could explain why the F-1 method only slightly outperforms the cross-entropy method, as the network's architecture compensates for variations in the loss function. Nonetheless, when testing on imbalanced data, F-1 method has a clear advantage. Like the reasoning from Logistic Regression model, F-1 method also focuses on minimizing FP and FN. 

When F1_method is tested on HW5_data, the very imbalanced data set, most of the time, the result for cross-entropy method often yields a higher F-1 score than F-1 method. I have tried to test Macro F-1 score and Cross-Entropy outputs a higher Macro F-1 score as well. These results provided a lot of feedback on the effectiveness and stability of F-1 method that could be improved. I think it is also important to understand the relation between hyper-parameter and model characteristic. I think it could work for a few reason: small batch size and moderate number of neurons in hidden layer prevent overfitting. Epochs is enough so help the model converge and 0.001 has just been a good learning rate for F-1 method to perform well, which Adam optimizer.
I have tried to apply PCA for data selection. This way, model only predict model based on relatively important features. However, it does not increase the performance of F-1 method dramatically. I have tried different batch size. And discovered that the performance of cross-entropy is not as easily influenced by batch-size as of F-1 method. 
Nonetheless, I discovered that F-1 method would outperforms tradition method every time by at least 15% when the data number is less (e.g. 7000 instead of 20000+). The gap decreases as the data number get larger. Thus, we can conclude that F-1 method is not stable when the data set is large. 
