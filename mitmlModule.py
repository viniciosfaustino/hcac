import numpy as np
from copy import deepcopy

class MITML:
    def __init__(self, _slack: float, iterations: int = 1):
        self.slack = _slack
        self.iterations = iterations

    def run(self, data: np.ndarray, A0: np.ndarray, similarity: dict, dissimilarity: dict):

        A = np.array(A0)
        lambda_s = np.zeros(len(similarity), dtype=np.float)
        lambda_d = np.zeros(len(dissimilarity), dtype=np.float)
        index_s = {pair: i for i, pair in enumerate(similarity)}
        index_d = {pair: i for i, pair in enumerate(dissimilarity)}

        slack_matrix_s = {}
        slack_matrix_d = {}

        i = 0
        while i < self.iterations:
            i += 1
            for pair in dissimilarity.keys():
                slack_matrix_d[pair] = deepcopy(dissimilarity[pair])

            for pair in similarity.keys():
                slack_matrix_s[pair] = deepcopy(similarity[pair])
            old_s = deepcopy(slack_matrix_s)
            old_d = deepcopy(slack_matrix_d)
            B = np.copy(A)
            A += self.calculate_matrix(slack_matrix_s, lambda_s, similarity, data, A, 1, index_s)
            A += self.calculate_matrix(slack_matrix_d, lambda_d, dissimilarity, data, A, -1, index_d)
            # print(B - A)
        return A

    def calculate_matrix(self, slack_matrix: dict, lambda_values: np.ndarray, pair_dict: dict, data: np.ndarray,
                         A: np.ndarray, delta: int, index: dict):
        for pair in pair_dict.keys():
            # 3.1
            x, y = pair
            dist = pair_dict[pair]
            xy = np.array([np.subtract(data[x], data[y])])

            # 3.2
            p = np.dot(np.dot(xy, A), xy.transpose())

            if p == 0:
                p = 0.001

            # 3.3
            # 3.4
            q = 0.001 if slack_matrix[pair] == 0. else slack_matrix[pair]
            alpha = min(lambda_values[index[pair]], (0.5 * delta * (1 / p - self.slack / q)))

            # 3.5
            beta = (delta * alpha) / (1 - delta * alpha * p)

            # 3.6
            q = 0.001 if self.slack + delta * alpha * slack_matrix[pair] == 0. else self.slack + delta * alpha * slack_matrix[pair]

            slack_matrix[pair] = (self.slack * slack_matrix[pair]) / q
            # print(slack_matrix[pair])

            # 3.7
            if alpha > 0.:
                print('pinto')
            lambda_values[index[pair]] = lambda_values[index[pair]] - alpha
            # 3.8
            # print(beta * A * dist * dist.transpose() * A)
            oldA = np.copy(A)
            # print(dist)
            # print(A)

            part1 = beta*np.multiply(A, dist)
            part2 = np.multiply(part1, dist.transpose())
            part3 = np.multiply(part2, A)

            # if beta > 0:
                # print('1:',part1)
                # print('2:',part1)
                # print('3:', part3)

                # print(part3)

        return part3

