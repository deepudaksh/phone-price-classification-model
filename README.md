# phone price classification model

Supervised Model For Mobile Phone Price Classification



Support vector machine classifier is one of the most popular machine learning classification algorithm. Svm classifier mostly used in addressing multi-classification problems.
In short: Multi-classification problem means having more that 2 target classes to predict.

Implementation is done Using Python language and sklearn libraries. 
Phone price classification dataset is having 20 features of phones from different brand and one target class.

These 20 features are:
•	battery_power	
•	blue	
•	clock_speed
•	dual_sim	
•	fc	
•	four_g	
•	int_memory	
•	m_dep	
•	mobile_wt	
•	n_cores	
•	pc	
•	px_height	
•	px_width	
•	ram	
•	sc_h	
•	sc_w	
•	talk_time	
•	three_g	
•	touch_screen	
•	wifi

Target Class:
The price_range type is the target class and it having 4 types

•	0
•	1
•	2
•	3

The idea of implementing svm classifier in Python is to use the phone specification features to train an SVM classifier and use the trained SVM model to predict the price_range.

Approach:
•	Training csv is split into Training data(80%) and Test data(20%) to build the model. 
•	We check the accuracy of the model after learning from training data.
•	Accuracy of the model by Confusion Matrix and Classification Report.
•	Test data is fed to classify the data into 4 categories of price_range.
