import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inputs):
    """
    Calculate the sigmoid for the given inputs (array)
    """
    sigmoid_scores = [1 / float(1 + np.exp(1 - x)) for x in inputs]
    return sigmoid_scores

def line_graph(x, y, x_title, y_title):
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

graph_x = range(-5, 10)
graph_y = sigmoid(graph_x)

line_graph(graph_x, graph_y, "Inputs", "Sigmoid Scores")
