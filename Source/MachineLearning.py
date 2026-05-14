from Types import Coefficients
from LearningDataGenerator import generate_learning_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def make_poly_features(df, suffix='1'):
    h = df[f'h{suffix}'].values
    s = df[f's{suffix}'].values
    a = df[f'a{suffix}'].values
    b = df[f'b{suffix}'].values
    z = df[f'z{suffix}_0'].values   

    features = {}
    features['z'] = z 
    features['h*z'] = h * z
    features['s*z'] = s * z
    features['b*z'] = b * z
    features['a*z'] = a * z
    features['h^2*z'] = (h**2) * z
    features['s^2*z'] = (s**2) * z
    features['b^2*z'] = (b**2) * z
    features['a^2*z'] = (a**2) * z
    features['h*s*z'] = h * s * z
    features['h*b*z'] = h * b * z
    features['h*a*z'] = h * a * z
    features['s*b*z'] = s * b * z
    features['s*a*z'] = s * a * z
    features['a*b*z'] = a * b * z

    features['h*z^2'] = h * z**2
    features['s*z^2'] = s * z**2
    features['a*z^2'] = a * z**2
    features['b*z^2'] = b * z**2
    features['z^2'] = z**2

    features['h*z^3'] = h * z**3
    features['a*z^3'] = a * z**3
    features['b*z^3'] = b * z**3
    features['s*z^3'] = s * z**3
    features['z^3'] = z**3




    return pd.DataFrame(features)

def get_data_from_file(path : str, test_part : int) -> tuple:
    column_names = ['Winner ID', 'h1', 'h2', 's1', 's2', 'a1', 'a2', 'b1', 'b2', 'z1_0', 'z2_0']
    data = pd.read_csv(path, sep="|", skipinitialspace=True, comment='#', names=column_names, header=None)
    
    poly1 = make_poly_features(data, suffix='1')
    poly2 = make_poly_features(data, suffix='2')
    X_diff = poly1 - poly2 
    
    scaler = StandardScaler()
    X_diff = pd.DataFrame(
        scaler.fit_transform(X_diff),
        columns=X_diff.columns
    )
    y = data['Winner ID'].values
    
    X_train = X_diff.iloc[test_part:]
    y_train = y[test_part:]
    X_test = X_diff.iloc[:test_part]
    y_test = y[:test_part]
    
    return X_train, y_train, X_test, y_test

def convert_data(learning_data : list[tuple[bool, Coefficients]], test_part : int) -> tuple:
    data_rows = []
    
    for ld in learning_data:
        row = {
            'Winner ID' : ld[0], 'h1' : ld[1].h1, 'h2' : ld[1].h2, 's1' : ld[1].s1, 's2' : ld[1].s2, 
            'a1' : ld[1].a1, 'a2' : ld[1].a2, 'b1' : ld[1].b1, 'b2' : ld[1].b2, 'z1_0' : ld[1].z1_0, 'z2_0' : ld[1].z2_0
        }
        data_rows.append(row)
    
    data = pd.DataFrame(data_rows)
    
    poly1 = make_poly_features(data, suffix='1')
    poly2 = make_poly_features(data, suffix='2')
    X_diff = poly1 - poly2 
    
    scaler = StandardScaler()
    X_diff = pd.DataFrame(
        scaler.fit_transform(X_diff),
        columns=X_diff.columns
    )
    y = data['Winner ID'].values
    
    X_train = X_diff.iloc[test_part:]
    y_train = y[test_part:]
    X_test = X_diff.iloc[:test_part]
    y_test = y[:test_part]
    
    return X_train, y_train, X_test, y_test
    
def get_lambdas(X_train, y_train, X_test, y_test):
    model = LinearSVC(C=1.0, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность на тестовых данных: {accuracy:.3f}")
    lambdas = model.coef_[0]
    return model, lambdas
    
def plot_lambdas(lambdas):
    feature_names = [
        "z", "h*z", "s*z", "b*z", "a*z",
        "h^2*z", "s^2*z", "b^2*z", "a^2*z",
        "h*s*z", "h*b*z", "h*a*z", "s*b*z", "s*a*z", "a*b*z",
        "h*z^2", "s*z^2", "a*z^2", "b*z^2", "z^2",   
        "h*z^3", "s*z^3", "a*z^3", "b*z^3", "z^3"
    ]

    colors = ['steelblue' if v >= 0 else 'tomato' for v in lambdas]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_names, lambdas, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Значение λ")
    plt.title("Коэффициенты модели (λ)")
    plt.tight_layout()
    plt.savefig("Plots/lambdas.png", dpi=150)
    plt.show()   
    
def plot_decision_boundary_pca(model, X_train, y_train):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train)

    model_2d = LinearSVC(C=1.0, random_state=42, max_iter=10000)
    model_2d.fit(X_2d, y_train)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z = model_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=0, alpha=0.2, colors=['tomato', 'steelblue'])
    plt.contour(xx, yy, Z, levels=0, colors='black', linewidths=1.5) 
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='black', linewidths=0.8, linestyles='--')

    plt.scatter(X_2d[y_train == 0, 0], X_2d[y_train == 0, 1], c='tomato', s=10, alpha=0.5) # why not == -1?
    
    plt.scatter(X_2d[y_train == 1, 0], X_2d[y_train == 1, 1], c='steelblue', s=10, alpha=0.5)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Граница решения SVM (PCA 2D)")
    plt.tight_layout()
    plt.savefig("Plots/decision_boundary_pca.png", dpi=150)
    plt.show()
    

def run_learning(num_data : int = 1000, max_time : float = 500, test_num : int = 500):
    X_train, y_train, X_test, y_test = convert_data(generate_learning_data(num_data, max_time), test_num)
    model, lambdas = get_lambdas(X_train, y_train, X_test, y_test)
    return X_train, y_train, model, lambdas

def get_learning_lambdas(num_data : int = 1000, max_time : float = 500, test_num : int = 500):
    X_train, y_train, model, lambdas = run_learning(num_data, max_time, test_num)
    return lambdas

if __name__ == "__main__":
    os.makedirs("Plots", exist_ok=True)
    # X_train, y_train, X_test, y_test = get_data_from_file("Data/LearningData2.txt", 500)
    
    X_train, y_train, model, lambdas = run_learning(1000, 500, 500)
    
    print("Найденные коэффициенты λ:")

    feature_names = [
        "z", "h*z", "s*z", "b*z", "a*z",
        "h^2*z", "s^2*z", "b^2*z", "a^2*z",
        "h*s*z", "h*b*z", "h*a*z", "s*b*z", "s*a*z", "a*b*z",
        "h*z^2", "s*z^2", "a*z^2", "b*z^2", "z^2",   
        "h*z^3", "s*z^3", "a*z^3", "b*z^3", "z^3"
    ]

    for i, (name, lam) in enumerate(zip(feature_names, lambdas), 1):
        print(f"λ_{i:2d}  {name:5s} = {lam:.6f}")

    plot_lambdas(lambdas=lambdas)
    plot_decision_boundary_pca(model, X_train, y_train)