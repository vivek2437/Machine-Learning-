import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv('/Users/ASUS/PycharmProjects/JupyterProject1/titanic (1)/train.csv')

test=pd.read_csv('/Users/ASUS/PycharmProjects/JupyterProject1/titanic (1)/test.csv')

test_passenger_ids = test['PassengerId']

# Add Survived column to test (for concatenation)
test['Survived'] = np.nan

# Combine datasets
full_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# Preview
full_data.head()
full_data.info()
full_data.describe()
full_data.isnull().sum()
full_data['Sex'].value_counts()
full_data['Cabin'].value_counts()
full_data['Embarked'].value_counts()
full_data['Pclass'].value_counts(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=full_data['Sex'],data=full_data)
plt.xlabel("Sex")
plt.ylabel("Counts")
plt.legend

sns.countplot(x=full_data['Pclass'],data=full_data)
plt.xlabel("Sex")
plt.ylabel("Counts")
plt.legend

# Title from Name
full_data['Title'] = full_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
full_data['Title'] = full_data['Title'].replace(['Lady', 'Countess','Capt','Col','Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
full_data['Title'] = full_data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Family Size and IsAlone
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
full_data['IsAlone'] = (full_data['FamilySize'] == 1).astype(int)

# Deck from Cabin
full_data['Deck'] = full_data['Cabin'].str[0]
full_data['Deck'] = full_data['Deck'].fillna('U')  # Unknown

# Ticket Group Size (count how many have the same ticket)
ticket_counts = full_data['Ticket'].value_counts()
full_data['TicketGroupSize'] = full_data['Ticket'].map(ticket_counts)

# Fare Binning
full_data['FareBin'] = pd.qcut(full_data['Fare'], 4, labels=False)

# Age Binning (temporarily fill NA with median to bin; we’ll impute better later)
age_median = full_data['Age'].median()
full_data['AgeBin'] = pd.qcut(full_data['Age'].fillna(age_median), 4, labels=False)

# Sex x Pclass interaction
full_data['Sex_Pclass'] = full_data['Sex'].astype(str) + "_" + full_data['Pclass'].astype(str)

full_data[['Name', 'Title', 'FamilySize', 'IsAlone', 'Deck', 'TicketGroupSize', 'FareBin', 'AgeBin', 'Sex_Pclass']].head()

# Fill Embarked with mode
full_data['Embarked'] = full_data['Embarked'].fillna(full_data['Embarked'].mode()[0])

# Fill Fare based on Pclass and Embarked groups
full_data['Fare'] = full_data.groupby(['Pclass', 'Embarked'])['Fare'].transform(lambda x: x.fillna(x.median()))

from sklearn.ensemble import RandomForestRegressor

# Features to predict Age
age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize']

# Encode categorical features temporarily
age_df = full_data[age_features + ['Age']].copy()
age_df['Sex'] = age_df['Sex'].map({'male': 0, 'female': 1})
age_df['Embarked'] = age_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
age_df['Title'] = age_df['Title'].astype('category').cat.codes

# Split into known and unknown Age
known_age = age_df[age_df['Age'].notnull()]
unknown_age = age_df[age_df['Age'].isnull()]

# Train RF Regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(known_age.drop('Age', axis=1), known_age['Age'])

# Predict missing Age
predicted_ages = rfr.predict(unknown_age.drop('Age', axis=1))

# Fill Age back
full_data.loc[full_data['Age'].isnull(), 'Age'] = predicted_ages

# full_data['Deck'] = full_data['Cabin'].str[0].fillna('U')
full_data['NumCabins'] = full_data['Cabin'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

import re

def extract_cabin_num(cabin):
    if pd.isnull(cabin):
        return 0
    numbers = re.findall(r'(\d+)', cabin)
    return int(numbers[0]) if numbers else 0

full_data['CabinNumber'] = full_data['Cabin'].apply(extract_cabin_num)
full_data[['Cabin', 'Deck', 'NumCabins', 'CabinNumber']].head(10)
full_data.drop(columns=['Cabin'], inplace=True)
full_data.isnull().sum()
full_data['Fare'] = full_data['Fare'].fillna(full_data['Fare'].median())
full_data['FareBin'].isnull().sum()



# Use rank to avoid duplicate bin edges
fare_ranks = full_data['Fare'].rank(method='min')

# Create equal-frequency Fare bins using rank
full_data['FareBin'] = pd.qcut(fare_ranks, 4, labels=False)


# In[ ]:


full_data['FareBin'].isnull().sum()


# In[ ]:


full_data.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Label Encode binary/categorical fields
le = LabelEncoder()
for col in ['Sex', 'Title', 'Deck', 'Sex_Pclass']:
    full_data[col] = le.fit_transform(full_data[col])

# One-hot encode Embarked
full_data = pd.get_dummies(full_data, columns=['Embarked'], drop_first=True)


# In[ ]:


drop_cols = ['PassengerId', 'Name', 'Ticket']
full_data.drop(columns=drop_cols, inplace=True)


# In[ ]:


# Split back into train/test
train_final = full_data[full_data['Survived'].notnull()].copy()
test_final = full_data[full_data['Survived'].isnull()].copy()

# Separate target variable
X = train_final.drop('Survived', axis=1)
y = train_final['Survived'].astype(int)

# For prediction later
X_test = test_final.drop('Survived', axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data into training and validation sets (e.g., 80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[ ]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
}

# Train and evaluate each model
for name, model in models.items():
    print("="*60)
    print(f"Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Accuracy
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


logreg_params = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__solver': ['liblinear', 'lbfgs']
}

logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

logreg_grid = GridSearchCV(logreg_pipe, logreg_params, cv=5, scoring='accuracy')
logreg_grid.fit(X_train, y_train)
print("Best Logistic Regression:", logreg_grid.best_params_)

knn_params = {
    'clf__n_neighbors': list(range(3, 21, 2)),
    'clf__weights': ['uniform', 'distance']
}

knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
print("Best KNN:", knn_grid.best_params_)

svm_params = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf', 'poly'],
    'clf__gamma': ['scale', 'auto']
}

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True))
])

svm_grid = GridSearchCV(svm_pipe, svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train, y_train)
print("Best SVM:", svm_grid.best_params_)

tree_params = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=5, scoring='accuracy')
tree_grid.fit(X_train, y_train)
print("Best Decision Tree:", tree_grid.best_params_)
nb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GaussianNB())
])

nb_pipe.fit(X_train, y_train)  # No tuning needed

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    "Logistic Regression": logreg_grid.best_estimator_,
    "KNN": knn_grid.best_estimator_,
    "SVM": svm_grid.best_estimator_,
}

for name, model in models.items():
    print("="*60)
    print(f"Model: {name}")
    y_pred = model.predict(X_val)

    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('svm', svm_grid.best_estimator_),
        ('logreg', logreg_grid.best_estimator_),
        ('knn', knn_grid.best_estimator_)
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_val)

print("Voting Classifier Accuracy:", accuracy_score(y_val, y_pred_voting))
print("Classification Report:\n", classification_report(y_val, y_pred_voting))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_voting))

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define base learners
base_learners = [
    ('svm', svm_grid.best_estimator_),
    ('logreg', logreg_grid.best_estimator_),
    ('knn', knn_grid.best_estimator_)
]

# Define meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Create the stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=False
)

# Fit on training data
stacking_clf.fit(X_train, y_train)

# Predict on validation set
y_pred = stacking_clf.predict(X_val)

# Evaluate
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

get_ipython().system('pip install lightgbm')

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

boosting_models = {
    'LightGBM': LGBMClassifier(random_state=42),
}

for name, model in boosting_models.items():
    print("="*60)
    print(f"Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

from sklearn.model_selection import cross_val_score

for name, model in boosting_models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define full stacking base learners
base_learners = [
    ('logreg', logreg_grid.best_estimator_),
    ('svm', svm_grid.best_estimator_),
    ('knn', knn_grid.best_estimator_),
    ('lgbm', LGBMClassifier(random_state=42))
]

# Meta learner
meta_learner = LogisticRegression(max_iter=1000)

# Final StackingClassifier
final_stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=False
)

# Fit
final_stack.fit(X_train, y_train)

# Predict
y_pred = final_stack.predict(X_val)

# Evaluate
print("FINAL STACKED MODEL RESULTS")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

test=pd.read_csv('/Users/ASUS/PycharmProjects/JupyterProject1/titanic (1)/test.csv')

# Save PassengerId BEFORE merging or dropping columns
test_passenger_ids = test['PassengerId'].copy()

submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions.astype(int)
})
submission.to_csv('submission.csv', index=False)
