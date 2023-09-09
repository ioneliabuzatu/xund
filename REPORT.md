for me:
In the context of medical text, NER is related to diseases, treatments, symptoms, etc. one word -> one class



# Report

the data is unbalanced, 80%+ represents class 'O'. this is shown in the final confusion matrix on the test data. solution: when splitting the data, do it in way that each class is faily represented in train and test splits, so having all classes equally present in the train set. when we look at the confusion matrix the model is clearly biased in classifying most of the samples as class 'O'. If the data cannot be balanced, possible solutions are, look up more data, consider data augumentation. 