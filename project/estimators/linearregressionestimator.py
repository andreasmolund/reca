from sklearn import linear_model
import numpy as np
import reservoir.util as rutil


class LinearRegressionEstimator:

    def __init__(self, n_output_nodes):
        self.output_nodes = [linear_model.LinearRegression()] * n_output_nodes

    def fit(self, X, y):
        for i, output_node in enumerate(self.output_nodes):
            output_node.fit(X, [labels[i] for labels in y])

    def predict(self, X):
        predictions = [None] * len(self.output_nodes)
        for i, output_node in enumerate(self.output_nodes):
            predictions[i] = rutil.classify_output(output_node.predict(X))
        return np.transpose(predictions, (1, 0))


