
import numpy as np
import matplotlib.pyplot as plt

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    """
    softmax_scores = np.exp(inputs) / float(sum(np.exp(inputs)))
    return softmax_scores

def line_graph(x, y, x_title, y_title):
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

graph_x = range(0, 9)
graph_y = softmax(graph_x)

line_graph(graph_x, graph_y, "Inputs", "Softmax Scores")