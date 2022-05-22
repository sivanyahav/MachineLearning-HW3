#!/usr/bin/env python
# coding: utf-8

from itertools import combinations
from numpy import log as ln
from numpy import exp as e
import pandas as pd
import matplotlib.pyplot as plt
import random

""" =====================================
    ============ Point Class ============
    ===================================== """


class Point:

    # init point
    def __init__(self, x=0., y=0., label=0.):
        self.x = float(x)  # coordinate x
        self.y = float(y)  # coordinate y
        self.label = int(label)  # label of the point
        self.weight = 0  # weight of the point


""" =====================================
    ============ Rule Class ============
    ===================================== """


class Rule:
    # init line with two points
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        self.m = 0 if point1.x == point2.x else (point1.y - point2.y) / (
                point1.x - point2.x)  # it is the slope of the line
        self.n = point1.y - (self.m * point1.x)  # find the n int the equation y=mx+n

    def eval_label(self, point):
        """
        this function receive point and the line
        the line predict the label of the point
        """
        if self.m != 0:  # if the rule is not a line parallel to the Y axis
            rule = self.m * point.x + self.n
            return -1 if point.y >= rule else 1

        else:  # if the rule is a line parallel to the Y axis
            return 1 if point.x >= self.point1.x else -1


# the best line is the line with the minimum error on the classify point
def get_best(points):
    comb = combinations(points, 2)
    best_line = Rule(Point(), Point())

    min_sum = 1  # min errors sum
    for pt in comb:
        new_rule = Rule(pt[0], pt[1])
        temp_sum = 0
        for x in points:
            # we check if the label of the point is the same or not to the label that the rule predicts
            # we check the points for which the line predicted false
            classify = int(x.label != new_rule.eval_label(x))

            temp_sum += x.weight * classify  # Update classifier weight based on error
                                             # if the point label is equal to the predicted label
                                             # we will get zero and mult the weight by 0.
                                             # else (classify == false)
                                             # we will mult in 1 and get the weight to calculate the error.

        if temp_sum < min_sum and temp_sum != 0:
            min_sum = temp_sum
            best_line = new_rule
    return best_line, min_sum  # return best line and his error


points = []


# get data from the file four_circle.txt
def insert_data():
    with open('four_circle.txt') as file:
        while True:
            lines = file.readline()
            if not lines:
                break
            data = lines.split(' ')
            point = Point((float(data[0])), (float(data[1])), (float(data[2])))
            points.append(point)
    return points


# This function receives a list of data points and splits it to 2 equal groups (train & test).
def split_data(points: list):
    if len(points) % 2 == 0:
        return points[:int(len(points) / 2)], points[int(len(points) / 2):]
    else:
        return points[:int((len(points) + 1) / 2)], points[int((len(points) + 1) / 2):]


def cal_weight(points, iter):
    """
    this function modify the weight of the points
    """

    best_lines = []
    for x in points:
        x.weight = 1 / len(points)
    for t in range(iter):
        best, errorLine = get_best(points)
        if errorLine > 0.5:
            print("error too big:", errorLine)
            break
        alpha_t = 0.5 * ln((1 - errorLine) / errorLine)
        Z = 0
        for x in points:
            x.weight = x.weight * e(-alpha_t * best.eval_label(x) * x.label)
            Z += x.weight
        for x in points:
            x.weight = x.weight / Z
        best_lines.append((best, alpha_t))
    return best_lines

test_plot = []
train_plot = []

# algorithm adaboost As in the presentation of the lesson
def run(iter=8, rounds=100, points_set=None):
    if points_set is None:
        points_set = []
    global test, train
    s = ""
    for k in range(1, iter + 1):
        tot_testing = 0
        tot_training = 0
        for i in range(rounds):
            random.shuffle(points_set)
            train, test = split_data(points)

            w = cal_weight(train, k)
            for x in test:
                H = 0
                for best_line, error in w:
                    H += error * best_line.eval_label(x)
                tot_testing += int(H * x.label < 0)

            for x in train:
                H = 0
                for best_line, error in w:
                    H += error * best_line.eval_label(x)
                tot_training += int(H * x.label < 0)
        test_errors = (tot_testing / rounds) / len(test)
        train_errors = (tot_training / rounds) / len(train)

        test_plot.append(test_errors)
        train_plot.append(train_errors)

        test_str = ", Test: [e(H(k)) on test: %.3f" % test_errors + "]"
        training_str = " Train: [e(H(k)) on train: %.3f" % train_errors + "]"

        s += "\nk = "
        s += str(k)
        s += training_str
        s += test_str
        print("k = ", k, training_str, test_str)
    return s

def plot_err(epochs, train_err, test_err):
    plt.xlabel('epochs')
    plt.ylabel('prediction error')
    plt.plot(epochs, train_err, color='pink', linewidth=3, label='train error')
    plt.plot(epochs, test_err, color='black', linewidth=3, label='test error')
    plt.legend()
    plt.show()

""" =====================================
    =============== Main ================
    ===================================== """
if __name__ == '__main__':
    data = insert_data()
    file = open("Output.txt", "w+")
    result = run(points_set=data)
    file.write(result)
    file.close()
    plot_err([*range(1, 9)], train_plot, test_plot)
