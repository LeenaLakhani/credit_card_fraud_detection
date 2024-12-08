#!/usr/bin/env python
# coding: utf-8

#                               Credit Card Fraud Detection Predictive Models 

# Introduction

# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation.
# 
# Due to confidentiality issues, there are not provided the original features and more background information about the data.
# 
# Features V1, V2, ... V28 are the principal components obtained with PCA;
# The only features which have not been transformed with PCA are Time and Amount. Feature Time contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature Amount is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.
# Feature Class is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# #Loading packages

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)





# # Reading the data

# In[2]:


data_df = pd.read_csv("creditcard.csv")


# # checking the data

# In[3]:


print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])


# # Glimpse the data

# We start by looking to the data features (first 5 rows).
# 
# 

# In[4]:


data_df.head()


# Let's look into more details to the data.

# In[5]:


data_df.describe()


# Looking to the Time feature, we can confirm that the data contains 284,807 transactions, during 2 consecutive days (or 172792 seconds).

# In[6]:


#checkmisssing values
data_df.isnull().sum()


# There is no missing value in the entire dataset.

# #  Data unbalanced

# In[7]:


import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot


# In[8]:


fraud_check=pd.value_counts(data_df['Class'],sort=True)
fraud_check.plot(kind='bar',rot=0,color='b')
plt.title("Legit and Fraud Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
labels=['Legit','Fraud']
plt.xticks(range(2),labels)
plt.show()


# Only 492 (or 0.172%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.

# # Data Exploration

# Transactions in time

# class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
# class_1 = data_df.loc[data_df['Class'] == 1]["Time"]
# 
# hist_data = [class_0, class_1]
# group_labels = ['Not Fraud', 'Fraud']
# 
# fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
# fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
# iplot(fig, filename='dist_only')

# Fraudulent transactions have a distribution more even than valid transactions - are equaly distributed in time, including the low real transaction times, during night in Europe timezone.
# 
# Let's look into more details to the time distribution of both classes transaction, as well as to aggregated values of transaction count and amount, per hour. We assume (based on observation of the time distribution of transactions) that the time unit is second.

# In[9]:


data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
df.head()


# In[10]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show();


# In[11]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show();


# In[12]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Average Amount of Transactions")
plt.show();


# In[13]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Max", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Max", data=df.loc[df.Class==1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show();


# In[14]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show();


# In[15]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show();


# # Outliers treatment

# We are not performing any outliers treatment for this particular dataset. Because all the columns are already PCA transformed, which assumed that the outlier values are taken care while transforming the data.

# In[16]:


sns.boxplot(data_df['Class'])
plt.show()
print('Percent of fraud transaction: ',len(data_df[data_df['Class']==1])/len(data_df['Class'])*100,"%")
print('Percent of normal transaction: ',len(data_df[data_df['Class']==0])/len(data_df['Class'])*100,"%")


# In[17]:


# Lets analyse the continuous values by creating histograms to understand the distribution
data=data_df.copy()
data.drop(columns='Class', inplace = True)

for feature in data.columns:
 
    data_df[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# Normalization is important in PCA since it is a variance maximizing exercise. It projects your original data onto directions which maximize the variance.
# 
# since V1, V2, … V28 are the principal components obtained with PCA we can clearly see that from above plot that from V1 to V28 variables are normalized. Variable 'Time' and 'Amount' is not normalized.

# In[18]:


#figure_factory module contains dedicated functions for creating very specific types of plots
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# 
# Credit Card Transactions Time Density Plot visualises the distribution of 'Not Fraud' and 'Fraud' transaction over a continuous interval or time period.
# 
# Fraudulent transactions have a distribution more even than valid transactions - are equaly distributed in time.
# 
# So 'Time' feature can't tell whether the trasaction is Fraudulent transactions or not.

# # Feature correlation

# In[19]:


plt.figure(figsize = (14,14))
plt.title('Feature correlation')
corr = data_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()


# Correlation is Positive when the values increase together, and Correlation is Negative when one value decreases as the other increases
# 
# Correlation can have a value:
# 
# 1 is a perfect positive correlation (between 'Amount' and V7, 'Amount' And 'V20')
# 0 is no correlation (between features V1-V28)
# -1 is a perfect negative correlation (between 'Time' and V3, 'Amount' and V2, 'Amount' and V5)

# # Outliers

# In[20]:


# Transaction amount 
data=data_df.copy()
data.drop(columns=['Class'], inplace = True)
for i in data.columns:
  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
  s = sns.boxplot(ax = ax1, x="Class", y=i, hue="Class",data=data_df, palette="PRGn",showfliers=True)
  s = sns.boxplot(ax = ax2, x="Class", y=i, hue="Class",data=data_df, palette="PRGn",showfliers=False)
  plt.show();


# Above Box plot say fradulant transaction has more outlier than non fradulant transaction. Since our data is highly imbalance and we have less amount frudulant trasaction so traforming outlier leads to loss of infomation. we will use outlier as it is.

# In[21]:


#Features density plot
col = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
       'V28']

i = 0
t0 = data_df.loc[data_df['Class'] == 0]
t1 = data_df.loc[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,30))

for feature in col:
    i += 1
    plt.subplot(7,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0", color='b')
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1", color='r')
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# Observation:
# 
# V3, V4, V10, V11, V17-V19 have clearly separated distributions for Class values 0 and 1
# V1, V2, V7, V9, V12, V14, V16, V18 have partially saperated distribution for Class 0 and 1
# V13, V15, ,V20, V22-V28 have almost similar distribution for Class 0 and 1
# V5, V6, V8, V21 have quite similar distribution for Class 0 and 1
# In general, with just few exceptions (Time and Amount), the features distribution for legitimate transactions (values of Class = 0) is centered around 0, sometime with a long queue at one of the extremities. In the same time, the fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distribution.

# In[22]:


pca_vars = ['V%i' % k for k in range(1,29)]
plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=t0[pca_vars].skew(), color='darkgreen')
plt.xlabel('Column')
plt.ylabel('Skewness')
plt.title('V1-V28 Skewnesses for Class 0')


# In[23]:


plt.figure(figsize=(12,4), dpi=80)
sns.barplot(x=pca_vars, y=t1[pca_vars].skew(), color='darkgreen')
plt.xlabel('Column')
plt.ylabel('Skewness')
plt.title('V1-V28 Skewnesses for Class 1')


# Above observation states that fradulant transaction has more skewness than normal transaction

# In[24]:


sns.set_style("whitegrid")
sns.FacetGrid(data_df, hue="Class", height = 6).map(plt.scatter, "Time", "Amount").add_legend()
plt.show()


# Observations
# 
# From the above two plots it is clearly visible that there are frauds only on the transactions which have transaction amount approximately less than 2500. Transactions which have transaction amount approximately above 2500 have no fraud.
# As per with the time, the frauds in the transactions are evenly distributed throughout time.

# In[25]:


FilteredData = data_df[['Time','Amount', 'Class']]
countLess = FilteredData[FilteredData['Amount'] < 2500]
countMore = data_df.shape[0] - len(countLess)
percentage = round((len(countLess)/data_df.shape[0])*100,2)
Class_1 = countLess[countLess['Class'] == 1]
print('Total number for transaction less than 2500 is {}'.format(len(countLess)))
print('Total number for transaction more than 2500 is {}'.format(countMore))
print('{}% of transactions having transaction amount less than 2500' .format(percentage))
print('{} fraud transactions in data where transaction amount is less than 2500' .format(len(Class_1)))


# In[26]:


sns.boxplot(x = "Class", y = "Amount", data = data_df)
plt.ylim(0, 5000)
plt.show()


# Observations:
# 
# There are 284358 transactions which has a transaction amount less than 2500. Means 99.84% of transactions have transaction amount less than 2500
# total number of fraud transactions in whole data are 492. It has been calculated that total number of fraud transactions in data where transaction amount is less than 2500 is also 492. Therefore, all 100% fraud transactions have transaction amount less than 2500 and there is no fraud transaction where transaction amount is more than 2500.
# From above box plot we can easily infer that there are no fraud transactions occur above the transaction amount of 2500. All of the fraud transactions have transaction amount less than 2500. However, there are many transactions which have a transaction amount greater than 2500 and all of them are genuine.

# In[27]:


Amount_0 = data_df.loc[data_df['Amount'] == 0]
print(Amount_0['Class'].value_counts())


# 
# There are 1,825 transactions that has 0 amount, 27 of them are fraud. One of the observation is 0 pending charge by a person is a verification method to verify the fraud.

# # Data Transformation

# In[28]:


from sklearn.preprocessing import StandardScaler, RobustScaler
data1=data_df.copy()
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data_df['scaled_amount'] = std_scaler.fit_transform(data_df['Amount'].values.reshape(-1,1))
data_df['scaled_time'] = std_scaler.fit_transform(data1['Time'].values.reshape(-1,1))

data_df.drop(['Amount', 'Time'], axis=1, inplace = True)
scaled_amount = data_df['scaled_amount']
scaled_time = data_df['scaled_time']

data_df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data_df.insert(0, 'scaled_amount', scaled_amount)
data_df.insert(1, 'scaled_time', scaled_time)
print(data_df.head())


# In[29]:


from sklearn.model_selection import train_test_split
X = data_df.drop(['Class'], axis=1)
Y = data_df['Class']
# This is explicitly used for with data imbalance
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X.shape, Y.shape)


# In[30]:


print('X train shape: ', X_train.shape)
print('X test shape: ', X_test.shape)
print('y train shape: ', y_train.shape)
print('y test shape: ', y_test.shape)


# In[31]:


print(y_test.value_counts())


# Handling Data Imbalance

# In[32]:


# Import necessary libraries
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Checking the class distribution
print("### Step 1: Checking the class distribution")
class_distribution = data_df['Class'].value_counts()
print("Class Distribution:\n", class_distribution)

# Visualizing the imbalance
class_distribution.plot(kind='bar', title='Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Step 2: Undersampling with RandomUnderSampler
print("\n### Step 2: Applying RandomUnderSampler")
X = data_df.drop(columns=['Class'])
y = data_df['Class']

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Checking new class distribution after undersampling
print("Class distribution after undersampling:\n", pd.Series(y_resampled).value_counts())

# Step 3: Oversampling with SMOTE
print("\n### Step 3: Applying SMOTE")
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)

# Checking new class distribution after oversampling
print("Class distribution after oversampling (SMOTE):\n", pd.Series(y_resampled_smote).value_counts())

# Step 4: Combined Approach with SMOTETomek
print("\n### Step 4: Applying SMOTETomek")
smotetomek = SMOTETomek(random_state=42)
X_resampled_combined, y_resampled_combined = smotetomek.fit_resample(X, y)

# Checking class distribution after SMOTETomek
print("Class distribution after SMOTETomek:\n", pd.Series(y_resampled_combined).value_counts())

# Step 5: Train and evaluate the model using undersampled data
print("\n### Step 5: Training and evaluating the model using undersampled data")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# # Hyperparameter Tunning

# In[35]:


get_ipython().system('pip install scikit-learn xgboost optuna')


# In[36]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
param_grid = {'n_estimators': [10, 50], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)


# # Model Building

# Model training with Entire dataset¶
# 
# Now we're ready to build machine learning models to predict whether a transaction is fraudulent. We'll train the following models:
# 
# Logistic regression
# Support vector classifier
# Desicision Tree
# Random forest
# Bagging classifier

# In[53]:


# Classifier Libraries
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.metrics import make_scorer, precision_score, recall_score, confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score, accuracy_score, average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  f_classif
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier


# In[54]:


classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "BaggingClassifier": BaggingClassifier(n_estimators=10, random_state=0),
    "SGDClassifier" : SGDClassifier(),
    "GradientBoostingClassifier" : GradientBoostingClassifier(),
    "xgb" : XGBClassifier()
}


# In[55]:


def plot(df):
  fraud = df[df['class']==1]
  normal = df[df['class']==0]
  fraud.drop(['class'],axis=1,inplace=True)
  normal.drop(['class'],axis=1,inplace=True)
  fraud = fraud.set_index('classifier')
  normal = normal.set_index('classifier')
  plt.figure(figsize = (8,2))
  sns.heatmap(fraud.iloc[:, :], annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"),linewidth=2)
  plt.title('class 1')
  plt.show()
  plt.figure(figsize = (8,2))
  sns.heatmap(normal.iloc[:, :], annot=True, cmap=sns.light_palette((210, 90, 60), input="husl"),linewidth=2)
  plt.title('class 0')
  plt.show()


# In[56]:


def roc_curve(y_test, rdict):
  sns.set_style('whitegrid')
  plt.figure()
  i=0
  fig, ax = plt.subplots(4,2,figsize=(16,30))
  for key,val in rdict.items():
    fpr, tpr, thresholds = metrics.roc_curve( y_test, val,
                                                  drop_intermediate = False )
    auc_score = metrics.roc_auc_score( y_test, val)
    i+= 1
    plt.subplot(4,2,i)
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title(key)
    plt.legend(loc="lower right")
  plt.show()


# In[57]:


def training(models, x, y, x_t, y_t):
    conf = []
    comp = []
    rdict = {}
    for key, model in models.items():
      model = model.fit(x, y)
      y_pred = model.predict(x_t)
      rdict[key] = y_pred
      tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
      precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_t, y_pred)
      r1 = {'Classifier': key, 'TN': tn, 'TP': tp, 'FN': fn, 'FP': fp}
      conf.append(r1)
      MCC = matthews_corrcoef(y_t, y_pred)
      AUROC = roc_auc_score(y_t, y_pred)
      Cohen_kappa = cohen_kappa_score(y_t, y_pred)
      accuracy = metrics.accuracy_score(y_t, y_pred)
      r2 = {'classifier': key,'matthews_corrcoef':MCC,'Cohen_kappa':Cohen_kappa,'accuracy': accuracy,'AUROC':AUROC, 'precision': precision[0],'recall':recall[0],'f1':fscore[0], 'class':0}
      r3 = {'classifier': key,'matthews_corrcoef':MCC,'Cohen_kappa':Cohen_kappa,'accuracy': accuracy,'AUROC':AUROC, 'precision': precision[1],'recall':recall[1],'f1':fscore[1], 'class':1}
      comp.append(r2)
      comp.append(r3)
    r11 = (pd.DataFrame(conf).to_markdown())
    r12 = pd.DataFrame(comp)
    print(f'\n\nRoc curve \n\n')
    roc_curve(y_t, rdict)
    print(f'\n\n confusion matrixs comparison \n\n')
    print(r11)
    print(f'\n\n Performance comparison \n\n')
    plot(r12)
    


# In[58]:


training(classifiers, X_train, y_train, X_test, y_test)


# # Model Training with only selected feature

# In[59]:


bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 
featureScores_df = featureScores.sort_values(['Score', 'Specs'], ascending=[False, True])  #naming the dataframe columns
print(featureScores_df)


# In[ ]:


col = ['V17', 'V14', 'V12','V10','V16','V3','V7','V11','V4','V18','V1','V9','V5','V2','V6','V21','V19','V20','V8','V27','scaled_time','V28','V24']


# In[60]:


training(classifiers, X_train[col], y_train, X_test[col], y_test)


# # Conclusion
# 
# * The performance of the model slightly increased after removing the few parameters like Amount, V13, V15, V22, V23, V25.
# * XGB classifier performed much better than the rest all the algorithm without any hyperparameter tweaking!

# In[61]:


xgb = XGBClassifier()
# X_train[col], y_train, X_test[col], y_test
xgb.fit(X_train[col],y_train)
y_pred_final = xgb.predict(X_test[col])


# In[62]:


import joblib
# Step 8: Model Deployment Plan
print("\n### Step 8: Model Deployment Plan")
# Save the trained model
model_filename = "fraud_detection_model.pkl"
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")


# In[ ]:




