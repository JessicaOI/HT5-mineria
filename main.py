import warnings
import io
from js import fetch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

async def load_dataset():
    res = await fetch('https://jupyterlite.anaconda.cloud/b0df9a1c-3954-4c78-96e6-07ab473bea1a/files/iris/iris.csv')
    csv_data = await res.text()
    dataset = pd.read_csv(io.StringIO(csv_data))
    dataset = dataset.drop('Id', axis=1)
    dataset.columns = ['Sepal-length', 'Sepal-width', 'Petal-length', 'Petal-width', 'Species']
    return dataset


def summarize_dataset(dataset):
    print("Dataset shape:", dataset.shape)
    print("\nFirst 10 rows:\n", dataset.head(10))
    print("\nLast 10 rows:\n", dataset.tail(10))
    print("\nBasic statistics:\n", dataset.iloc[:, 1:].describe())
    print("\nClass distribution:\n", dataset.groupby('Species').size())

def visualize_dataset(dataset):
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()

    dataset.hist()
    plt.show()

    scatter_matrix(dataset)
    plt.show()

def evaluate_algorithms(X_train, Y_train):
    models = [
        ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC(gamma='auto'))
    ]

    results = []
    names = []

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

def make_predictions(X_train, Y_train, X_validation, Y_validation):
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    print("Accuracy:", accuracy_score(Y_validation, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))
    print("\nClassification Report:\n", classification_report(Y_validation, predictions))

async def main():
    dataset = await load_dataset()
    summarize_dataset(dataset)
    visualize_dataset(dataset)

    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    print("\nEvaluating Algorithms:")
    evaluate_algorithms(X_train, Y_train)

    print("\nMaking predictions and evaluating them:")
    make_predictions(X_train, Y_train, X_validation, Y_validation)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

