import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

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

    return pd.DataFrame(features)

def get_data(path : str, test_part : int) -> tuple:
    column_names = ['Winner ID', 'h1', 'h2', 's1', 's2', 'a1', 'a2', 'b1', 'b2', 'z1_0', 'z2_0']
    data = pd.read_csv(path, sep="|", skipinitialspace=True, comment='#', names=column_names, header=None)
    poly1 = make_poly_features(data, suffix='1')
    poly2 = make_poly_features(data, suffix='2')
    X_diff = poly1 - poly2 
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
    return lambdas
    
    
if __name__ == "__main__":
    
    X_train, y_train, X_test, y_test = get_data("Data/LearningData.txt", 500)

    lambdas = get_lambdas(X_train, y_train, X_test, y_test)

    print("Найденные коэффициенты λ:")

    feature_names = [
        "z", "h*z", "s*z", "b*z", "a*z",
        "h^2*z", "s^2*z", "b^2*z", "a^2*z",
        "h*s*z", "h*b*z", "h*a*z", "s*b*z", "s*a*z", "a*b*z"
    ]

    for i, (name, lam) in enumerate(zip(feature_names, lambdas), 1):
        print(f"λ_{i:2d}  {name:5s} = {lam:.6f}")
