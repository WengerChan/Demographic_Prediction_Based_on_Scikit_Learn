# Demographic Prediction Based on Scikit-learn

WengerChan


In the information-developed mobile Internet era, demographics has a very large role in many actual business scenarios, such as ad placement, recommendation systems, and the selection of sites under certain brands. The mobile user's behavior habits often reflect the user's demographic attributes. 

This article will be based on Python's powerful machine learning module Scikit-learn, through the modeling of a large number of behavioral information classification algorithms, and ultimately achieve predicting demographic through the mobile user behavior habits. 

This thesis uses the data about 200,000 users provided by TalkingData. The data consists of two parts: one part is gender-age information and it is divided into 12 groups, the other part is the user's behavioral attributes, including time and geographic coordinate information, and mobile phone. The brand and model, the APP installed and used, and the category of the app. This article is based on the above data modeling, and finally realizes that the mobile end user's demographic attributes are predicted according to the behavioral habits. 

This article is based on open source software Python and NumPy, SciPy, Pandas and Scikit-learn build the environment, the main machine learning methods used are Logistic Regression and Support Vector Machine. In the modeling phase, 3-fold cross validation was used. The evaluation index of the model is loss function. This article mainly performed data preprocessing, feature engineering, model training and testing, and logloss evaluation, the output and evaluation, etc. This paper provides logistic regression and SVM model implementation methods, as well as specific practical processes, and finally gives the logloss and the optimal model after training. Through specific practices, as far as the models involved in this paper are concerned, after the data reaches an order of magnitude, the SVM algorithm performs very well. The logarithm loss predicted by the model remains below 2.5, but it is a logistic regression algorithm in terms of time. Several times or even dozens of times. In practical applications, certain trade-offs may be made based on accuracy requirements and time requirements.
