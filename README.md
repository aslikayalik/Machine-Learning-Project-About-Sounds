# Machine-Learning-Project-About-Sounds

###	ORIGINAL DATASET REGRESSION PROJECT
1.1	DATASET AND PREPROCESSING

•	Data Source: https://sound-effects.bbcrewind.co.uk/

•	Data Description: Dataframe has 3 columns and 376 rows. Column names : file, folder, label. The types of columns : file is object, folder is int64, label is object. I extract durations and sizes of sounds and merge them to make a new dataframe.

•	ML Problem definition: Estimating the duration of an audio file based on its size.

•	Data Split: In this code, the dataset  and the target variable are split into three parts using the train_test_split function: train, test, and validation.
The X and y data are split into test data and the remaining data as train data, with a test size of 0.2 specified by the test_size parameter. The random_state parameter is used for reproducibility and is set to 42 to ensure the same random split is obtained when the code is run again.
In the second line, the X_train and y_train data are again split using the train_test_split function, this time into validation data and the remaining data as train data, with a test size of 0.2 specified by the test_size parameter. The random_state parameter is again set to 42.
As a result:
X_train contains 64% of the train data,
X_test contains 20% of the test data,
X_val contains 16% of the validation data.
The same proportions apply to y_test, y_train, and y_val.

•	Data Preprocessing: I first converted the mp3 files I downloaded into wav files(mp3_to_wav.py) , and then I found the duration of each file using the get_audio_duration() function in a for loop. I performed the same process for the size of the audio files as well, using the format_size() function.

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/b7c80b5b-8887-4238-b3e1-df3964170135)

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/9f0558a8-d12d-4fcd-ad68-521aee3b9265)

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/6ba2b817-15a5-49a4-9124-c88f9cc5d028)

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/7fed84c3-4d97-4b4d-a55d-9909e075d509)

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/c37670d5-7587-49ac-985b-ac9c0852ef59)



 



1.2	REGRESSION STEPS

- After obtaining the sizes and durations(I used coreR for this and I imported coreR.py file into all of my regression models) , merge them together to create a new dataframe.
- Proceed to split this dataframe into test, validation, and train sets. 
- Create an instance of model
- Create a dictionary parameter grid of values (for the ElasticNet)
- Create a GridSearchCV object and run a grid search for the best parameters for the model based on the scaled training data (for the ElasticNet).
- Display the best combination of parameters for the model.
- Evaulate the model’s performance on the unseen test set.
- Calculate MAE, MSE and R^2

1.3	REGRESSION RESULTS AND MODEL SELECTION

I used the linear regression model for regression and then tried the modified version of it called Elastic Net.

Linear Regression :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/fe4e3585-68df-47e3-aa55-4e5f9afec298)

 

Elastic Net:
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/80ee3cc6-70f4-491f-98de-eb0bdf844bc5)



I attempted to compare Elastic Net and Linear regression by examining the metrics of mean_absolute_error, mean_squared_error and R2. Based on these results, constructing and predicting the model using Elastic Net is more appropriate as it yields the lowest value for mean_absolute_error and mean_squared_error.


I added the size of an audio file from external sources to the model and asked it to predict the duration.
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/585a259b-4d5d-47ba-bcb8-bf55fa2aa5d5)

 
 
###	ORIGINAL DATASET CLASSIFICATION PROJECT
2.1	DATASET AND PREPROCESSING

•	Data Source: https://sound-effects.bbcrewind.co.uk/

•	Data Description: Dataframe has 3 columns and 376 rows. Column names : file, folder, label. The types of columns : file is object, folder is int64, label is object. I extract features of sound and merge the sounds and lanels together to make a new dataframe.


•	ML Problem definition: The task is to classify the dataset based on predefined labels.

•	Data Split: In this code, the dataset  and the target variable are split into three parts using the train_test_split function: train, test, and validation.
The X and y data are split into test data and the remaining data as train data, with a test size of 0.2 specified by the test_size parameter. The random_state parameter is used for reproducibility and is set to 42 to ensure the same random split is obtained when the code is run again.
In the second line, the X_train and y_train data are again split using the train_test_split function, this time into validation data and the remaining data as train data, with a test size of 0.2 specified by the test_size parameter. The random_state parameter is again set to 42.
As a result:
X_train contains 64% of the train data,
X_test contains 20% of the test data,
X_val contains 16% of the validation data.
The same proportions apply to y_test, y_train, and y_val.

•	Data Preprocessing : I first converted the mp3 files I downloaded into wav files(I mentioned that in the regression part), and then I found the features of each file using the features_extractor() function in a for loop. 

2.2	CLASSIFICATION STEPS

- After obtaining the features(I used core file for this and I imported core.py file into all of my classification models), merge features and labels together to create a new dataframe.
- Proceed to split this dataframe into test, validation, and train sets. 
- Create an instance of default model
- Train and test the model
- Evaulate the model’s performance on the unseen test set.
- Calculate validation accuracy, test accuracy , precision, recall , f1-score. 
- Display confusion matrix
- Predict an unseen data according to the model you choose

2.3	CLASSIFICATION RESULTS AND MODEL SELECTION

•	I used Support Vector Machines, K-Nearest Neighbors, Decision Tree, Linear Discriminant Analysis, Naive Bayes
•	
Support Vector Machines :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/e86cddb1-2b91-458c-8ec4-74f7a0cc399b)



K-Nearest Neighbors :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/d32c393d-73df-43e4-aa9c-94167cd56dee)


 

Decision Tree :

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/cd84630e-5eb7-4456-94b2-f582e50e4fc3)



Linear Discriminant Analysis :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/4a0f5f4d-b55c-47fa-b026-f51703501a93)




Naive Bayes
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/60c659d9-6313-406c-b818-59615535a658)


 
Support Vector machines model is the best model according to the comparison table. I loaded the external file ‘07070171.wav’ into the model and asked it to predict which class it belonged to. It was a sound of a clock and the SVC model correctly predicted it.

![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/4e2a4e29-112a-484d-9b6c-71efafec851e)


 


 
###	ORIGINAL DATASET CLUSTERING PROJECT

3.1	DATASET AND PREPROCESSING

•	Data Source: https://sound-effects.bbcrewind.co.uk/

•	Data Description: Dataframe has 3 columns and 376 rows. Column names : file, folder, label. The types of columns : file is object, folder is int64, label is object. I extract features of sound and merge the sounds and labels together to make a new dataframe.


•	ML Problem definition: Cluster similar audio files together.

•	Data Split:  

Split dataset into features(X) and labels(y)
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['label'].tolist())
X, y = make_blobs(n_samples=376, centers=5, cluster_std=0.6, random_state=0)

•	Data Preprocessing: I first converted the mp3 files I downloaded into wav files(I mentioned that in the regression part), and then I found the features of each file using the features_extractor() function in a for loop. 


3.2	CLUSTERING STEPS
- After obtaining the features, merge features and labels together to create a new dataframe.
- Split the dataframe 
- Create an instance of default model
- Train and test the model
- Evaulate the model’s performance on the unseen test set(using adjusted_rand_score)
- Plotting the model


3.3	CLUSTERING RESULTS AND MODEL SELECTION

I used agglomerative clustering model and K-means model.

Agglomerative Clustering :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/fe756c40-4456-4446-8d1e-66a8f5a664e3)

 

Agglomerative clustering adjusted_rand_score :
 ![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/7d4dc5b1-ea00-4e83-9019-eda07bed0de6)




K-Means Clustering :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/43408ec3-6c01-46d1-b427-01d681d10cdc)

 

K-means clustering adjusted_rand_score :
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/fb6fc2a6-3f7f-4164-87a2-0e8ff303c72d)

 


I attempted to make predictions using the K-means model. I provided an audio file from an external source and extracted its features, resulting in 40 features. However, the prediction process gave the following error : “ ValueError: X has 40 features, but KMeans is expecting 2 features as input.” To fix this error, I tried reducing the number of features by applying Principal Component Analysis (PCA), but still encountered the same error. I was unable to perform the prediction process in the clustering. 
![image](https://github.com/aslikayalik/Machine-Learning-Project-About-Sounds/assets/96055823/ceeabb71-9b69-41fe-a847-ee229ead9e5d)

 


