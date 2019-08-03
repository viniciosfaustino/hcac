import numpy as np
from copy import deepcopy

class MITML:
    def __init__(self, slack: float, iterations: int = 1):
        self.iterations = iterations

    def run(self, data: np.ndarray, A0: np.ndarray, X: np.ndarray, similarity: dict, dissimilarity: dict):

        A = np.array(A0)
        lambda_s = np.zeros(len(similarity)).transpose()
        lambda_d = np.zeros(len(dissimilarity)).transpose()

        slack_matrix_s = {}
        slack_matrix_d = {}
        i = 0
        while i < self.iterations:
            i += 1
            for pair in dissimilarity.keys():
                slack_matrix_s[pair] = deepcopy(dissimilarity[pair])

            for pair in similarity.keys():
                slack_matrix_d[pair] = deepcopy(similarity[pair])

            # for pair in similarity.keys():
            A = self.calculate_matrix(slack_matrix_s, lambda_s, similarity, data, A, 1)

            # for pair in dissimilarity.keys():
            A = self.calculate_matrix(slack_matrix_d, lambda_d, dissimilarity, data, A, -1)

            for pair in similarity.keys():
                #3.1
                x, y = pair
                dist = similarity[pair]
                xy = np.subtract(data[x], data[y])
                #3.2
                p = np.dot(np.dot(xy.transpose(), A), xy)

                if p == 0:
                    p = 0.01

                #3.3
                delta = 1
                #3.4
                alpha = min(lambda_s[pair], (0.5*delta*(1/p - slack/slack_matrix_s[pair])))
                #3.5
                beta = (delta*alpha)/(1 - delta*alpha*p)
                #3.6
                slack_matrix_s[pair] = (slack*slack_matrix_s)/(slack + delta*alpha*slack_matrix_s[pair])
                #3.7
                lambda_s[pair] = lambda_s[pair] - alpha
                #3.8
                A = A + beta*A*dist*dist.transpose()*A

            for pair in dissimilarity.keys():
                #3.1
                x, y = pair
                dist = dissimilarity[pair]
                xy = np.subtract(data[x], data[y])
                #3.2
                p = np.dot(np.dot(xy.transpose(), A), xy)

                if p == 0:
                    p = 0.01

                #3.3
                delta = -1
                #3.4
                alpha = min(lambda_d[pair], (0.5*delta*(1/p - slack/slack_matrix_d[pair])))
                #3.5
                beta = (delta*alpha)/(1 - delta*alpha*p)
                #3.6
                slack_matrix_d[pair] = (slack*slack_matrix_d[pair])/(slack + delta*alpha*slack_matrix_d[pair])
                #3.7
                lambda_d[pair] = lambda_d[pair] - alpha
                #3.8
                A = A + beta*A*dist*dist.transpose()*A

    def calculate_matrix(self, slack_matrix: dict, lambda_values: np.ndarray, pair_dict: dict, data: np.ndarray, A: np.ndarray, delta: int):
        for pair in pair_dict.keys():
            # 3.1
            x, y = pair
            dist = pair_dict[pair]
            xy = np.subtract(data[x], data[y])
            # 3.2
            p = np.dot(np.dot(xy.transpose(), A), xy)

            if p == 0:
                p = 0.01

            # 3.3
            delta = 1
            # 3.4
            alpha = min(lambda_values[pair], (0.5 * delta * (1 / p - self.slack / slack_matrix[pair])))
            # 3.5
            beta = (delta * alpha) / (1 - delta * alpha * p)
            # 3.6
            slack_matrix[pair] = (self.slack * slack_matrix) / (self.slack + delta * alpha * slack_matrix[pair])
            # 3.7
            lambda_values[pair] = lambda_values[pair] - alpha
            # 3.8
            A = A + beta * A * dist * dist.transpose() * A

            return A
        pass

