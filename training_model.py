import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Pour sauvegarder le modèle

def train_model():
    # 1. Chargement des données
    iris = datasets.load_iris()
    
    # X = Longueur pétale, y = Largeur pétale
    X = iris.data[:, np.newaxis, 2]
    y = iris.data[:, 3]

    # 2. Séparation Entraînement / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Initialisation et Entraînement
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # 4. Évaluation rapide
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 5. Sauvegarde du modèle (Format .joblib)
    model_filename = "regression_iris.joblib"
    joblib.dump(regr, model_filename)

if __name__ == "__main__":
    train_model()