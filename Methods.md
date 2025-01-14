This section explains how to address the discrete nature of F-1 score 
function and transform it into a differentiable and continuous function 
for optimization purposes.
Methods
The probabilistic approximation of True Positive (TP), False Positive (FP), and False Negative (FN) is shown as below. 
 
 
TP= ∑_(k=1)^N▒〖y_k*P_k 〗
FP= ∑_(k=1)^N▒〖〖(1-y〗_k)*P_k 〗  

 
TN= ∑_(k=1)^N▒〖(1-y_k )*(1-p_k)〗
FN=∑_(k=1)^N▒〖y_k*(1-p_k)〗 
yk refers to the ground truth -- whether the actual label is 0 or 1.
xk refers to the given data. 
Pk = P(xk), refers to the probability that given xk the corresponding yk will equal to 1.
Likewise, 1-Pk refers to the probability that that given xk the corresponding yk be 0.
P(x) typically represented as sigmoid functions, because they map into values to a range between 0 and 1.
	These 4 expressions are very intuitive. When yk=1, and our prediction, Pk, approaches 1, predictions are close to the ground truth, we will maximize TP and minimize FN. Likewise when yk=0, and our prediction, Pk, approaches 0, we will maximize TN and minimize FP. When our prediction does not align with the ground truth, FN and FP will be maximized and TP and TN will be minimized. 
Using these functions to express Precision, Recall, and F-1 score:
Precision=TP/(TP+FP) 
Recall=TP/(TP+FN) 
F1 Score=2*(Precision * Recall )/(Precision +Recall) 
F1 Score (in terms of TP,FP,TN)=2*TP/(2*TP+FP+FN) 
It is important to notice: the F-1 score generated here should be output as the opposite of the value. When using gradient descent backpropagation, to minimize F-1 score loss is to maximize F-1 score. 
Since TP, FP, and FN are all approximated by probabilities, it is continuous differentiable, F1 score function will also be continuous and differentiable. Thus, we wo
![image](https://github.com/user-attachments/assets/d5a40724-872e-4110-b357-1cbc525647bd)
