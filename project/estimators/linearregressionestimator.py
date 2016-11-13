from sklearn import linear_model
from reservoir.util import classify_output


class LinearRegressionEstimator:

    def __init__(self, n_output_nodes):
        self.output_nodes = [linear_model.LinearRegression()] * n_output_nodes

    def fit(self, X, y):
        for i, output_node in enumerate(self.output_nodes):
            output_node.fit(X, [label[i] for label in y])

    def predict(self, X):
        predictions = [None] * len(self.output_nodes)
        for i, output_node in enumerate(self.output_nodes):
            predictions[i] = classify_output(output_node.predict(X))
        return predictions


