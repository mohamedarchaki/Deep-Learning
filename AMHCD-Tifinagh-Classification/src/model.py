import numpy as np
from src.activations import relu, relu_derivative, softmax
from src.optimizers import AdamOptimizer

class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, lambda_reg=0.0, optimizer='sgd'):
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, \
            "layer_sizes doit etre une liste avec au moins 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), \
            "Toutes les tailles doivent etre des entiers positifs"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, \
            "Le taux d'apprentissage doit etre positif"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.optimizer_type = optimizer.lower()
        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        if self.optimizer_type == 'adam':
            self.opt = AdamOptimizer(self.weights, self.biases, lr=self.learning_rate)

    def forward(self, X):
        assert isinstance(X, np.ndarray), "X doit etre un tableau numpy"
        assert X.shape[1] == self.layer_sizes[0], "Dimension d'entree incorrecte"

        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))

        z_out = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z_out)
        output = softmax(z_out)
        self.activations.append(output)
        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        loss = - (1.0 / m) * np.sum(y_true * np.log(y_pred))

        if self.lambda_reg > 0:
            l2_cost = (self.lambda_reg / (2.0 * m)) * sum(np.sum(np.square(w)) for w in self.weights)
            loss += l2_cost
        return loss

    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)

    def backward(self, X, y, outputs):
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        dZ = outputs - y
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i + 1].T) * relu_derivative(self.z_values[i])
            self.d_weights[i] = (self.activations[i].T @ dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

        if self.lambda_reg > 0:
            for i in range(len(self.weights)):
                self.d_weights[i] += (self.lambda_reg / m) * self.weights[i]

        # Application de la mise a jour
        if self.optimizer_type == 'adam':
            self.opt.update(self.weights, self.biases, self.d_weights, self.d_biases)
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * self.d_weights[i]
                self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0.0

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)

            num_batches = int(np.ceil(X.shape[0] / batch_size))
            train_loss = epoch_loss / num_batches

            train_pred = self.forward(X)
            train_acc = self.compute_accuracy(y, train_pred)

            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_acc = self.compute_accuracy(y_val, val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)