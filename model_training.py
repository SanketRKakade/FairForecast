import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Import the preprocess_data function
from preprocess import preprocess_data

def train_svm(X_train, y_train, X_test, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print(f'SVM Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(svm, 'models/svm_model.pkl')
    print('SVM model saved to models/svm_model.pkl')
    return svm

if __name__ == '__main__':
    ticker = 'AAPL'  # Same ticker as used in preprocess
    try:
        stock_data = pd.read_csv(f'stock_data/{ticker}.csv', index_col='Date', parse_dates=True)
        X_train, X_test, y_train, y_test = preprocess_data(stock_data)

        # Train the SVM model
        train_svm(X_train, y_train, X_test, y_test)
    except FileNotFoundError:
        print(f'File stock_data/{ticker}.csv not found. Ensure that data has been downloaded successfully.')
