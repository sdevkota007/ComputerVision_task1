import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    '''
    reads the given data file
    :param file_name: name of the file with full path
    :return: matrix A, and B
    '''
    with open(file_name, 'r') as file:
        lines = file.readlines()
        A = np.matrix([x.strip().split("    ") for x in lines[2:22]], dtype= np.float32)
        b = np.matrix([x.strip() for x in lines[26:46]], dtype= np.float32).reshape((20,1))
        return A,b


def mmse(A, b):
    '''
    calculates the least square solution for vector X
    :param A: a numpy matrix
    :param b: a numpy matrix
    :return: vector X (a numpy matrix) which is the least square solution
    '''
    X = (A.T * A).I * A.T * b
    e = A*X - b
    print("Mean Squared Error: {}".format((e.T*e).tolist()[0][0]))
    return X


def gradient_descent(A, b, precision = 0.000001, rate_of_decay= 0.01, max_iterations = 1000, plot = True):
    '''
    Calculates X, the solution to AX=B using iterative gradient descent
    :param A:
    :param b:
    :param precision: Desired precision of result
    :param rate_of_decay: Step size multiplier
    :param max_iterations: Maximum number of iterations
    :param plot: bool value, plots a graph of cost vs step if true
    :return: X, Solution to AX = b
    '''
    _,j = A.shape
    X = np.matrix(np.random.rand(j,1))
    J = 0
    list_of_cost = []
    for i in range(max_iterations):
        e = A * X - b
        J_prev = J
        J = (e.T * e).tolist()[0][0]      # cost of this iteration
        J_diff = J_prev - J               # difference in cost of previous iteration and current iteration

        del_J = 2 * A.T * e
        X = X - rate_of_decay * del_J

        list_of_cost.append(J)

        if abs(J_diff) <= precision:
            break

    print("Mean Squared Error: {}".format(J))
    if plot:
        plt.plot(list_of_cost)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost function")
        plt.title("Gradient Descent")
        plt.show()


    #print(i)
    return X





def main():
    X, b = read_data("data.txt")
    # print(X, X.shape)
    # print(b, b.shape)

    print("Least square error or MMSE solution: ")
    h_mmse = mmse(X,b)
    print(h_mmse)
    print("Finding solution using iterative gradient descent approach: ")
    h_gd = gradient_descent(X,b)
    print(h_gd)

if __name__ == "__main__":
    main()