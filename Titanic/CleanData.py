
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random as rnd


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[5]:


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv') 
combine = [train_df, test_df]


# In[6]:


train_df.head()


# In[7]:


train_df.columns


# In[8]:


train_df.info()
print("_"*40)
test_df.info()


# In[9]:


train_df.shape


# In[10]:


train_df.describe()


# In[11]:


train_df.describe(include=['O'])


# In[12]:


train_df[['Pclass','Survived']].groupby("Pclass",as_index = False).mean().sort_values(by = "Survived",ascending = False)


# In[13]:


train_df[['Sex',"Survived"]].groupby("Sex",as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[14]:


train_df[['SibSp','Survived']].groupby("SibSp",as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[15]:


train_df[['Parch','Survived']].groupby("Parch",as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[16]:


g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,'Age',bins=20)


# In[17]:


grid = sns.FacetGrid(train_df,col='Survived',row="Pclass",size=2.2,aspect=1.6)
grid.map(plt.hist,"Age",bins=20,alpha=0.5)
grid.add_legend()


# In[18]:


grid = sns.FacetGrid(train_df,col='Embarked',size=2.2,aspect=1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
grid.add_legend()


# In[19]:


grid = sns.FacetGrid(train_df,row="Survived",col="Embarked",size=2.2,aspect=1.6)
#ci 表示 置信度
grid.map(sns.barplot,"Sex",'Fare',alpha=0.5,ci=None)   
grid.add_legend()


# In[20]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[21]:


train_df['Name']


# In[22]:


for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
# pd.crosstab(train_df["Title"],train_df['Sex'])
train_df[["Age",'Title']].groupby(by="Title").mean()
train_df.head()


# In[23]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[24]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset["Title"].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# In[25]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# In[26]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0})
train_df.head()


# In[27]:


guess_ages = np.zeros((2,3))
guess_ages


# In[29]:


for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            age_guess = guess_df.median()
             # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
# train_df.isnull().any()
# train_df[train_df.isnull().values==True]


# In[278]:


train_df["AgeBand"] = pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[279]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# In[280]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[281]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[282]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[283]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# In[284]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[285]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[286]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[287]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[288]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[289]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[291]:


train_df.isnull().any()

