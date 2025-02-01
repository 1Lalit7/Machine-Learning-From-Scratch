import numpy as np

class LinearRegression:
    """
    A simple Linear Regression model trained using Gradient Descent.

    Attributes:
        learning_rate (float): The step size for gradient descent optimization.
        num_iterations (int): Number of iterations for training.
        l1_lambda (float): Strength of L1 regularization. Default is 0 (no Lasso).
        l2_lambda (float): Strength of L2 regularization. Default is 0 (no Ridge).
        elastic_net_ratio (float): Mixing parameter (0 = Ridge, 1 = Lasso, in-between = Elastic Net).
    """

    def __init__(self, 
                learning_rate: float = 0.01, 
                num_iterations: int = 100,
                l1_lambda: float = 0.0,
                l2_lambda: float = 0.0,
                elastic_net_ratio: float = 0.0,
                verbose: bool = False):
        """
        Initializes the Linear Regression model with hyperparameters.

        Args:
            learning_rate (float): The learning rate for gradient descent. Defaults to 0.01.
            num_iterations (int): Number of iterations for training. Defaults to 100.
            l1_lambda (float): L1 regularization strength (Lasso). Default is 0.
            l2_lambda (float): L2 regularization strength (Ridge). Default is 0.
            elastic_net_ratio (float): Mixing parameter for L1/L2 regularization (0 = Ridge, 1 = Lasso).
            verbose (bool): Whether to print progress messages during training.

        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.elastic_net_ratio = elastic_net_ratio
        self.verbose = verbose

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Squared Error (MSE) loss.

        Args:
            y_true (np.ndarray): True target values of shape (n_samples,).
            y_pred (np.ndarray): Predicted values of shape (n_samples,).

        Returns:
            float: The computed Mean Squared Error.
        """

        mse_loss = np.mean((y_true - y_pred) ** 2)
        if self.elastic_net_ratio:
            l1_penalty = self.elastic_net_ratio * self.l1_lambda * np.sum(np.abs(self.weights))  # L1 (Lasso)
            l2_penalty = (1 - self.elastic_net_ratio) * self.l2_lambda * np.sum(self.weights ** 2)  # L2 (Ridge)
        else:
            l1_penalty = self.l1_lambda * np.sum(np.abs(self.weights))  # L1 (Lasso)
            l2_penalty = self.l2_lambda * np.sum(self.weights ** 2)  # L2 (Ridge)
        return mse_loss + l1_penalty + l2_penalty

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the Linear Regression model using Batch Gradient Descent.

        Args:
            X (np.ndarray): Training feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.001
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.num_iterations):
            # Compute predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute error
            error = y_pred - y

            # Compute gradients
            dl_dw = (1 / n_samples) * np.dot(X.T, error)
            dl_db = (1 / n_samples) * np.sum(error)

            if self.elastic_net_ratio:
                # Apply Elastic Net regularization
                # L1 (Lasso)
                dl_dw += self.elastic_net_ratio * self.l1_lambda * np.sign(self.weights)  
                # L2 (Ridge)
                dl_dw += (1 - self.elastic_net_ratio) * 2 * self.l2_lambda * self.weights  
            else:
                # Apply L1/L2 regularization to gradient update
                # L1 gradient (Lasso)
                dl_dw += self.l1_lambda * np.sign(self.weights)  
                # L2 gradient (Ridge)
                dl_dw += 2 * self.l2_lambda * self.weights  

            # Update parameters
            self.weights -= self.learning_rate * dl_dw
            self.bias -= self.learning_rate * dl_db

            # Compute loss
            loss = self.mean_squared_error(y, y_pred)
            self.loss_history.append(loss)

            progress_msg = f"Epoch: [{epoch + 1}/{self.num_iterations}], Loss: {loss:.4f}"

            if self.verbose:
                print(progress_msg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Linear Regression model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the R-squared score of the model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).

        Returns:
            float: The computed R-squared score.
        """
        y_pred = self.predict(X)
        total_variance = np.var(y)
        explained_variance = np.var(y - y_pred)
        return 1 - (explained_variance / total_variance)


X = np.random.randn(100, 10)
y = np.random.randn(100)
linear_regression = LinearRegression(verbose=True)
linear_regression.fit(X, y)
y_pred = linear_regression.predict(X)
score = linear_regression.score(X, y)
# print(f"Predicted target values: {y_pred}")
print(f"R-squared score: {score}")

