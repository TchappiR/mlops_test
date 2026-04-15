import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    # Chargement et préparation [cite: 15, 16]
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

    # Entraînement [cite: 17]
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    # Évaluation [cite: 18, 22]
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Modèle entraîné avec une Accuracy de: {acc:.4f}")

    # Sauvegarde [cite: 19, 21]
    joblib.dump(model, "model.joblib")
    print("Modèle sauvegardé sous model.joblib")

if __name__ == "__main__":
    train()