from preprocessing.data_preprocessing import DataPreprocessing


filepath = "./data/winequality.csv"
numerical_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

dp = DataPreprocessing()
x = dp.data_ingestion(filepath)
x = dp.data_normalization(x, numerical_cols)
X_train, X_test, y_train, y_test = dp.data_split(x, "quality")
print(f'training data shape {X_train.shape}')
print(f'test data shape {X_test.shape}')