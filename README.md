# Disease-Prediction-Using-ML

## Abstract
#### This is a short project to predict the disease of someone, based on the symptoms that he/she possesses, by using supervised machine learning. I will use 3 different models and combine them for a more accurate prediction. The dataset was acquired from [Kaggle](Kaggle.com) which will be divided into two parts; for testing and training the models. This project follows the steps outlined in this [website](https://www.geeksforgeeks.org/disease-prediction-using-machine-learning/). Any modifications or extra functionality I will outline and document it accordingly. See project setup for python and package specification used.
------------------------------------

### Dataset

As mentioned, the dataset can be found in Kaggle named "[Prognosis Disease Symptoms](https://www.kaggle.com/datasets/noeyislearning/disease-prediction-based-on-symptoms/data)". It is stored in CSV format files with each row representing a patient record, 132 columns representing different symptoms plus one last column indicating the disease/prognosis. The 132 symptoms column are in binary format i.e if a patient record suffers from a specific symptom then (1) indicates the presence of that specific symptom. On the other hand, (0) indicates the absence of it. The prognosis column is a string type meaning it will have to be encoded for our models to assign a predicted disease.

### Models

All models used in this project come from sci-kit learn package. The three models I will be using are the following:

* Support Vector Classifier

  - This model is part of an bigger umbrella supervised ML called [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) (SVM) that are a set of methods used for classification, regression and outliers detection. As you may have guessed, I used the classification model or SVC. It is capable of performing binary and multi-class classification on a dataset. It takes two input arrays. The first has shape (n_samples, n_features) which holds the training samples, the second array has shape (n_samples) holding the class label. After both inputs are fitted, it can be used to predict new values. It essentially finds the optimal hyperplane in an N-dimensional space to separate data points into different classes. The algorithm maximizes the margin between the closest points of different classes. More info can about SVM can be found [here](https://www.geeksforgeeks.org/support-vector-machine-algorithm/?ref=gcse_outind).

* Naive Bayes Classifier

  - A ML algorithm used for classification tasks and it does this using Bayes' Theorem to find probabilities and by assuming the presence of one feature does not affect other features, hence the use of the word "Naive". In sci-kit learn there are a few different Naive Bayes Classifier to choose from. For this project I will use the [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) which assumes the likelihood of the features to be Gaussian. More infomation can be found [here](https://www.geeksforgeeks.org/naive-bayes-classifiers/).

* Random Forest Classifier

  - The Random Forest Classifier creates a set of decision trees from a randomly selected subset of the training set. It uses these sets of decision trees for a majority voting to decide on a final prediction. Random Forest Classification is an ensemble learning technique designed to enhance the accuracy and robustness of classification tasks. How it creates these subsets of decision trees is defined by employing a technique called bagging (Bootstrap Aggregating). This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers. The sci-kit page for this model can be found [here](https://scikit-learn.org/stable/modules/ensemble.html#forest) as well as a beginner friendly introduction [here](https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/?ref=gcse_outind).
 
------------------------------------

Finally, it is important to evaluate the performance of our models which is why I use the [Confusion Matrix](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/), a table that compares predictions given by a model to the actual results. It also is able to calculate key measures like accuracy, precision, and recall, which give a better idea of performance, especially when the data is imbalanced. The sci-kit page for the confusion matrix gives [instructions](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) on how to use it as well as providing examples.
