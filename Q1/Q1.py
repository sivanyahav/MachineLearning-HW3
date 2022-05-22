import numpy as np

points = []
l = []


def insert_data():
    with open('two_circle.txt') as file:
        while True:
            lines = file.readline()
            point = []
            if not lines:
                break
            data = lines.split(' ')
            point.append((float(data[0])))
            point.append((float(data[1])))
            l.append((float(data[2])))
            points.append(point)
    return points, l


def perceptron(X, y, n_iter=150):
    miss = 0
    misses_points = []
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    for i in range(n_iter):
        for idx, x_i in enumerate(X):
            guess = np.where(np.dot(x_i, weights) > 0, 1, -1)
            if (y[idx] - np.squeeze(guess)) != 0:  # mistake
                if y[idx] == 1:
                    weights += x_i
                else:
                    weights -=  x_i

                miss += 1
                misses_points.append([idx, np.array2string(x_i)])
                break

    return weights, miss


if __name__ == '__main__':
    points, l = insert_data()

    weights, miss = perceptron(np.array(points), l)
    print('miss:', miss, ' weights: ', weights)
    print('Error rates are: ', miss / len(points))
