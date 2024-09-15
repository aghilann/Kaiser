# california_housing.py
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple

def task(container_image: str = None):
    def decorator(func):
        func.container_image = container_image
        return func
    return decorator

def workflow(func):
    return func

@task(container_image="intel/classical-ml:latest-py3.10")
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_california_housing
    # Load the California housing dataset
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X, y

@task(container_image="intel/classical-ml:latest-py3.10")
def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # For simplicity, let's assume no preprocessing is needed
    import numpy as np
    return X, y

@task(container_image="intel/classical-ml:latest-py3.10")
def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    # Split data into train/test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

@task(container_image="intel/classical-ml:latest-py3.10")
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    from sklearn.linear_model import LinearRegression
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

@task(container_image="intel/classical-ml:latest-py3.10")
def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error
    import numpy as np
    # Evaluate the model: calculate mean squared error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return mse

@workflow
def california_housing_workflow() -> None:
    # Execute the workflow
    X, y = load_data()
    X_processed, y_processed = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    california_housing_workflow()