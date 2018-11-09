#%% [markdown]
# # Pima Indian Diabetes Prediction (with Model Reload)
#%% [markdown]
# Import some basic libraries.
# * Pandas - provided data frames
# * matplotlib.pyplot - plotting support
# 
# Use Magic %matplotlib to display graphics inline instead of in a popup window.
# 
#%%
# Disable warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#%%
import pandas as pd                 # pandas is a dataframe library
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data

get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# ## Loading and Reviewing the Data

#%%
df = pd.read_csv("./data/pima-data.csv")


#%%
df.shape


#%%
df.head(5)


#%%
df.tail(5)

#%% [markdown]
# ### Definition of features
# From the metadata on the data source we have the following definition of the features.
# 
# | Feature  | Description | Comments |
# |--------------|-------------|--------|
# | num_preg     | number of pregnancies         |
# | glucose_conc | Plasma glucose concentration a 2 hours in an oral glucose tolerance test         |
# | diastolic_bp | Diastolic blood pressure (mm Hg) |
# | thickness | Triceps skin fold thickness (mm) |
# |insulin | 2-Hour serum insulin (mu U/ml) |
# | bmi |  Body mass index (weight in kg/(height in m)^2) |
# | diab_pred |  Diabetes pedigree function |
# | Age (years) | Age (years)|
# | skin | ???? | What is this? |
# | diabetes | Class variable (1=True, 0=False) |  Why is our data boolean (True/False)? |
# 
#%% [markdown]
# ## Check for null values

#%%
df.isnull().values.any()

#%% [markdown]
# ### Correlated Feature Check
#%% [markdown]
# Helper function that displays correlation by color.  Red is most correlated, Blue least.

#%%
def plot_corr(df, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks


#%%
plot_corr(df)


#%%
df.corr()


#%%
df.head(5)

#%% [markdown]
# The skin and thickness columns are correlated 1 to 1.  Dropping the skin column

#%%
del df['skin']


#%%
df.head(5)

#%% [markdown]
# Check for additional correlations

#%%
plot_corr(df)

#%% [markdown]
# The correlations look good.  There appear to be no coorelated columns.
#%% [markdown]
# ## Mold Data
#%% [markdown]
# ### Data Types
# 
# Inspect data types to see if there are any issues.  Data should be numeric.

#%%
df.head(5)

#%% [markdown]
# Change diabetes from boolean to integer, True=1, False=0

#%%
diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

#%% [markdown]
# Verify that the diabetes data type has been changed.
#%% [markdown]
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      
#      

#%%
df.head(5)

#%% [markdown]
# ### Check for null values

#%%
df.isnull().values.any()

#%% [markdown]
# No obvious null values.
#%% [markdown]
# ### Check class distribution 
# 
# Rare events are hard to predict

#%%
num_obs = len(df)
num_true = len(df.loc[df['diabetes'] == 1])
num_false = len(df.loc[df['diabetes'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))

#%% [markdown]
# Good distribution of true and false cases.  No special work needed.
#%% [markdown]
# ### Spliting the data 
# 
# 70% for training, 30% for testing

#%%
from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values     # predictor feature columns (8 X m)
y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 
                            # test_size = 0.3 is 30%, 42 is the answer to everything

#%% [markdown]
# We check to ensure we have the the desired 70% train, 30% test split of the data

#%%
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

#%% [markdown]
# #### Verifying predicted value was split correctly

#%%
print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]), (len(df.loc[df['diabetes'] == 1])/len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]), (len(df.loc[df['diabetes'] == 0])/len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

#%% [markdown]
# ### Post-split Data Preparation
#%% [markdown]
# #### Hidden Missing Values

#%%
df.head()

#%% [markdown]
# Are these 0 values possible?
# 
# How many rows have have unexpected 0 values?

#%%
print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))

#%% [markdown]
# #### Impute with the mean

#%%
from sklearn.preprocessing import Imputer

#Impute with mean all 0 readings
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

#%% [markdown]
# ## Training Initial Algorithm - Naive Bayes

#%%
from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())

#%% [markdown]
# ### Performance on Training Data

#%%
# predict values using the training data
nb_predict_train = nb_model.predict(X_train)

# import the performance metrics library
from sklearn import metrics

# Accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print()

#%% [markdown]
# ### Performance on Testing Data

#%%
# predict values using the testing data
nb_predict_test = nb_model.predict(X_test)

from sklearn import metrics

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

#%% [markdown]
# #### Metrics

#%%
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))

#%% [markdown]
# ## Random Forest

#%%
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)      # Create random forest object
rf_model.fit(X_train, y_train.ravel()) 

#%% [markdown]
# ### Predict Training Data

#%%
rf_predict_train = rf_model.predict(X_train)
# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))

#%% [markdown]
# ### Predict Test Data

#%%
rf_predict_test = rf_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))


#%%
print(metrics.confusion_matrix(y_test, rf_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))

#%% [markdown]
# ## Logistic Regression

#%%
from sklearn.linear_model import LogisticRegression

lr_model =LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))

#%% [markdown]
# Setting regularization parameter

#%%
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(max_iter=500,solver='lbfgs', C=C_val, random_state=42)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
        
    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")

#%% [markdown]
# ### Logisitic regression with class_weight='balanced'

#%%
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(max_iter=500,solver='lbfgs',C=C_val, class_weight="balanced", random_state=42)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
        
    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")


#%%
from sklearn.linear_model import LogisticRegression
lr_model =LogisticRegression(max_iter=500,solver='lbfgs',class_weight="balanced", C=best_score_C_val, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))
print(metrics.recall_score(y_test, lr_predict_test))

#%% [markdown]
# ### LogisticRegressionCV

#%%
from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced")  # set number of jobs to -1 which uses all cores to parallelize
lr_cv_model.fit(X_train, y_train.ravel())

#%% [markdown]
# ### Predict on Test data

#%%
lr_cv_predict_test = lr_cv_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test)))
print(metrics.confusion_matrix(y_test, lr_cv_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_cv_predict_test))

#%% [markdown]
# # Using your trained Model
#%% [markdown]
# ## Save trained model to file

#%%
from sklearn.externals import joblib  
joblib.dump(lr_cv_model, "./data/pima-trained-model.pkl")

#%% [markdown]
# ## Load trained model from file

#%%
lr_cv_model = joblib.load("./data/pima-trained-model.pkl")

#%% [markdown]
# ## Test Prediction on data
# 
# Once the model is loaded we can use it to predict on some data.  In this case the data file contains a few rows from the original Pima CSV file.
# 

#%%
# get data from truncated pima data file
df_predict = pd.read_csv("./data/pima-data-trunc.csv")
print(df_predict.shape)


#%%
df_predict

#%% [markdown]
# The truncated file contained 4 rows from the original CSV.
# 
# Data is the same is in same format as the original CSV file's data.  Therefore, just like the original data, we need to transform it before we can make predictions on the data.  
# 
# Note: If the data had been previously "cleaned up" this would not be necessary.
# 
# We do this by executed the same transformations as we did to the original data
# 
# Start by dropping the "skin" which is the same as thickness, with different units.

#%%
del df_predict['skin']
df_predict

#%% [markdown]
# We need to drop the diabetes column since that is what we are predicting.  
# Store data without the column with the prefix X as we did with the X_train and X_test to indicate that it contains only the columns we are prediction.

#%%
X_predict = df_predict
del X_predict['diabetes']

#%% [markdown]
# Data has 0 in places it should not.  
# 
# Just like test or test datasets we will use imputation to fix this.

#%%
#Impute with mean all 0 readings
from sklearn.impute import SimpleImputer 
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
X_predict = fill_0.fit_transform(X_predict)

#%% [markdown]
# At this point our data is ready to be used for prediction.  
#%% [markdown]
# ## Predict diabetes with the prediction data.  Returns 1 if True, 0 if false

#%%
lr_cv_model.predict(X_predict)


