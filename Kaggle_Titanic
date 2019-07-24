import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#Load Data
train_df=pd.read_csv("train.csv")
test_df = pd.read_csv('test.csv')
#train_df.head()
#test_df.head()

# Check missing data
def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    total = total[total > 0]
    percent = percent[percent > 0]   
    return (total, percent)
    
[total_f, percent_f]=missingdata(train_df)
[predict_t, predict_p] = missingdata(test_df)
print("Missing data in Training dataset\n",total_f,'\n \n',percent_f)
print('----------------------------------------------------------------------')
print('\nMissing data in Test dataset\n',predict_t, '\n \n', predict_p)
print('----------------------------------------------------------------------')

# Data Cleaning
drop_column = ['Cabin']
train_df.drop(drop_column, axis=1, inplace = True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

drop_column = ['Cabin']
test_df.drop(drop_column, axis=1, inplace = True)
test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)


print('\nCheck the NaN value in train data')
print(train_df.isnull().sum())
print('\nCheck the NaN value in test data')
print(test_df.isnull().sum())
print('----------------------------------------------------------------------')

# One Hot Vectorization
train_df = pd.get_dummies(train_df, columns = ["Embarked"], prefix=["Em_type"])
train_df.head()

test_df = pd.get_dummies(test_df, columns = ["Embarked"], prefix=["Em_type"])
test_df.head()


# Feature Engineering
dataset = train_df
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

test_dataset = test_df
test_dataset['FamilySize'] = test_dataset['SibSp'] + test_dataset['Parch'] + 1


# Data Binning
dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,14,20,40,120], labels=['Children','Teenage','Adult','Elder'])
dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','Medium_fare', 'Average_fare','High_fare'])

test_dataset['Age_bin'] = pd.cut(test_dataset['Age'], bins=[0,14,20,40,120], labels=['Children','Teenage','Adult','Elder'])
test_dataset['Fare_bin'] = pd.cut(test_dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','Medium_fare', 'Average_fare','High_fare'])


# Extracting titles from names - unique identities
import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names in Training Dataset
dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
                                                                               
traindf=train_df
drop_column = ['Age','Fare','Name','Ticket']
traindf.drop(drop_column, axis=1, inplace = True)

traindf = pd.get_dummies(traindf, columns = ["Sex","Title","Age_bin","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Fare_type"])


# Create a new feature Title, containing the titles of passenger names in Test Dataset
test_dataset['Title'] = test_dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
test_dataset['Title'] = test_dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_dataset['Title'] = test_dataset['Title'].replace('Mlle', 'Miss')
test_dataset['Title'] = test_dataset['Title'].replace('Ms', 'Miss')
test_dataset['Title'] = test_dataset['Title'].replace('Mme', 'Mrs')
                                                                               
testdf=test_df
drop_column = ['Age','Fare','Name','Ticket']
testdf.drop(drop_column, axis=1, inplace = True)

testdf = pd.get_dummies(testdf, columns = ["Sex","Title","Age_bin","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Fare_type"])


#traindf.head()
#testdf.head()

# Data Correlation Matrix
sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()
print('----------------------------------------------------------------------')

# Machine Learning Model
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix

all_features = traindf.drop("Survived",axis=1)
Targeted_feature = traindf["Survived"]
X_predict = testdf

# Splitting the data - One part to train and other part to validate (test)
X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape,X_predict.shape

# Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='gini', n_estimators=700,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)
model.fit(X_train,y_train)
prediction_rm=model.predict(X_test)
result_test = model.predict(X_predict)

print('----------------------------The Accuracy of the model----------------------------')
print('The accuracy of the Random Forest Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_rm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')
print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)
sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)

# Implement the model on Test Data and predict the output (Survival data) and complete final submission
submission = pd.DataFrame({"PassengerId": X_predict.PassengerId,"Survived":result_test})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
