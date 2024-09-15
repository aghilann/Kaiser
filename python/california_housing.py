# simple_workflow.py

from typing import Tuple, List

# Define the @task and @workflow decorators
def task(func):
    # For this example, the decorator does nothing special
    return func

def workflow(func):
    return func

@task
def load_data() -> List[Tuple[float, float]]:
    # Simulate loading data: A list of tuples with (input, output) values
    data = [
        (1.0, 2.0),
        (2.0, 4.0),
        (3.0, 6.0),
        (4.0, 8.0),
        (5.0, 10.0),
    ]
    return data

@task
def preprocess_data(data: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    # Preprocess the data: Separate inputs (X) and outputs (y)
    X = [x for x, _ in data]
    y = [y for _, y in data]
    # Basic normalization (scaling) - dividing by the max value
    max_value = max(X)
    X_scaled = [x / max_value for x in X]
    return X_scaled, y

@task
def split_data(X: List[float], y: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
    # Split data into train/test sets (80/20 split)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train: List[float], y_train: List[float]) -> Tuple[float, float]:
    # Train a simple linear model: y = a * x + b
    # For simplicity, assume b = 0, so y = a * x
    # Calculate 'a' as the ratio of mean of y to mean of X
    a = sum(y_train) / sum(X_train) if sum(X_train) != 0 else 0
    return a, 0  # returns slope 'a' and intercept 'b'

@task
def evaluate_model(model: Tuple[float, float], X_test: List[float], y_test: List[float]) -> float:
    # Evaluate the model: calculate mean squared error
    a, b = model
    y_pred = [a * x + b for x in X_test]
    mse = sum((y_true - y_pred) ** 2 for y_true, y_pred in zip(y_test, y_pred)) / len(y_test)
    print(f"Mean Squared Error: {mse}")
    return mse

@workflow
def simple_data_processing_workflow() -> None:
    # Execute the workflow
    data = load_data()
    X_scaled, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    simple_data_processing_workflow()
