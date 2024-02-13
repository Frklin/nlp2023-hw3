import matplotlib.pyplot as plt
import numpy as np



def plot_images(head: list, tail: list, span: list):
    """
    Plot the head, tail, span matrices and save them
    """
    max_value = 400
    matrix_size = max_value + 1  

    # Initialize matrices for head, tail, span, and combined
    head_matrix = np.zeros((matrix_size, matrix_size))
    tail_matrix = np.zeros((matrix_size, matrix_size))
    span_matrix = np.zeros((matrix_size, matrix_size))

    # Populate the matrices
    for (i, j) in head:
        head_matrix[i-1, j-1] = 1
    for (i, j) in tail:
        tail_matrix[i, j] = 1
    for (i, j) in span:
        span_matrix[i, j] = 1

    # Combine all the matrices
    combined_matrix = head_matrix + tail_matrix + span_matrix

    # Plot and save the matrices
    plot_and_save_matrix(head_matrix, 'Head Relations', 'hw3/images/head.png', colorscale='Greens')
    plot_and_save_matrix(tail_matrix, 'Tail Relations', 'hw3/images/tail_relations.png', colorscale='Blues')
    plot_and_save_matrix(span_matrix, 'Span Relations', 'hw3/images/span_relations.png', colorscale='Purples')
    plot_and_save_matrix(combined_matrix, 'Combined Relations', 'hw3/images/combined_relations.png', colorscale='Reds')




def plot_and_save_matrix(matrix: np.ndarray, title: str, filename: str, colorscale: str = 'viridis'):
    """
    Plot and save the matrix
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=colorscale, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()
