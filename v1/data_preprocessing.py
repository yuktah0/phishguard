import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():

    # Load dataset
    df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\PhishGuard\v1\data\Phishing_Legitimate_full.csv")

    # Separate features and target
    X = df.drop(columns=["id", "CLASS_LABEL"])
    y = df["CLASS_LABEL"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=7000,
        test_size=3000,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)