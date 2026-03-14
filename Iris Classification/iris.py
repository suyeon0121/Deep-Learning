from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  
import numpy as np  

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris

def train_model(X_train, y_train):
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=200
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test, iris):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    return y_pred

def plot_confusion_matrix(model, X_test, y_test, iris):
    ConfusionMatrixDisplay.from_estimator(
        model, 
        X_test, 
        y_test,
        display_labels=iris.target_names
    )
    plt.title("Confusion Matrix")
    plt.show()

def plot_pca(X, y, iris):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure()

    for i, name in enumerate(iris.target_names):
        plt.scatter(
            X_pca[y==i, 0],
            X_pca[y==i, 1],
            label=name
        )
    
    plt.xlabel("PC1")  
    plt.ylabel("PC2")  
    plt.title("PCA Feature Visualization")
    plt.legend()
    plt.show()

def plot_decision_boundary(X, y):
    X_vis = X[:, [2, 3]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vis)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=200
    )
    model.fit(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1
    y_min, y_max = X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=y,
        edgecolor='k'
    )

    plt.xlabel("Petal Length(scaled)")  
    plt.ylabel("Petal Width(scaled)")   
    plt.title("Decision Boundary(Logistic Regression)")
    plt.show()

def main():
    X, y, iris = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_model(X_train_scaled, y_train)
    evaluate(model, X_test_scaled, y_test, iris)
    plot_confusion_matrix(model, X_test_scaled, y_test, iris)
    plot_pca(X, y, iris)
    plot_decision_boundary(X, y)

if __name__ == "__main__":
    main()
