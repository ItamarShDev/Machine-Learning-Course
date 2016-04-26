import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import copy

def plot():
    """
    Show the Plot and Refresh it
    Thanks to Alon Shmilo!
    """
    # plt.title(string)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.plot([0, 0, 1, 1], [0, 1, 0, 1], 'ro', ms=10)
    plt.axis([-1, 2, -1, 2])
    plt.plot([1], [1], 'bo', ms=10)
    plt.plot([-1, 2], Y)
    plt.plot

x_inputs = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
z_outputs = np.array([[1],[1],[1],[0]])
r = 0.1
z = 1
t = 0.5
i = 0
w = [0,0,0.05]
old_w = [0,0,0]
count = 0
if __name__ == "__main__":
    """
    This program mimics the neurons learning style
    This is done by calculating weights to each input to get the wanted output
    The program works until the weights stops to change
    At each step a curve will appear, presenting the current state
    """
    file = open("results.txt", 'w+')
    file.write("X0|X1|X2|Z|W0|W1|W2|C0|C1|C2|S |D |W0|W1|W2|\n")
    while(True):
        c = x_inputs[i, 0:3] * w
        s = np.sum(c)
        n = 0
        if s > t:
            n = 1
        e = int(z_outputs[i] - n)
        d = r*e
        old_w = copy.copy(w)
        w += x_inputs[i, 0:3] * d
        print 100*"-"
        print "X0 | X1 | X2 | Z | W0  | W1  | W2  | C0  | C1  | C2  |  S  |  D  | W0  | W1  | W2  |"
        print "%d  | %d  | %d  | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (x_inputs[i, 0], x_inputs[i, 1], x_inputs[i, 2], z_outputs[i], old_w[0], old_w[1], old_w[2], c[0], c[1], c[2], s, d, w[0], w[1], w[2] )
        file.write( "%d  | %d  | %d  | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |\n" % (x_inputs[i, 0], x_inputs[i, 1], x_inputs[i, 2], z_outputs[i], old_w[0], old_w[1], old_w[2], c[0], c[1], c[2], s, d, w[0], w[1], w[2]))
        y1 = (t-w[1]*(-1)-w[0])/w[2]
        y2 = (t-2*w[1]-w[0])/w[2]

        Y = [y1, y2]
        plot()
        pause(0.2)
        graph = plt.figure()
        plt.close(graph)
        plt.close()
        plt.show()


        if (old_w == w).all():
            count += 1
            if count == 3:
                plot()
                plt.show()
                file.close()
                print "DONE"
                break
        else:
            count = 0
        i = (i + 1) % 4
