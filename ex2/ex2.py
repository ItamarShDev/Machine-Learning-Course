import numpy as np
import matplotlib.pyplot as plt

x_inputs = [[1,0,0,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
r = 0.1
z = 1
t = 0.5
i = 0
w0 = 0
w1 = 0
w2 = 0
count = 0
if __name__ == "__main__":
    file = open("results.txt", 'w+')
    file.write("X0|X1|X2|Z|W0|W1|W2|C0|C1|C2|S |D |W0|W1|W2|\n")
    w0_c = w0
    w1_c = w1
    w2_c = w2
    while(True):
        w0_t = w0
        w1_t = w1
        w2_t = w2
        x0 = x_inputs[i][0]
        x1 = x_inputs[i][1]
        x2 = x_inputs[i][2]
        z = x_inputs[i][3]

        c0 = w0*x0
        c1 = w1*x1
        c2 = w2*x2
        s = c0 + c1 + c2
        n = 0
        if s > t:
            n = 1
        e = z - n
        d = r*e
        w0 += x0 * d
        w1 += x1 * d
        w2 += x2 * d

        print 100*"-"
        print "X0 | X1 | X2 | Z | W0  | W1  | W2  | C0  | C1  | C2  |  S  |  D  | W0  | W1  | W2  |"
        print "%d  | %d  | %d  | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (x0,x1,x2,z,w0_t,w1_t,w2_t,c0,c1,c2,s,d,w0,w1,w2)
        file.write("%d  | %d  | %d  | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |\n" % (x0,x1,x2,z,w0_t,w1_t,w2_t,c0,c1,c2,s,d,w0,w1,w2))

        if w0_c == c0 and w1_c == c1 and w2_c == c2:
            count += 1
        if count == 2:
            file.close()
            break
        w0_c = c0
        w1_c = c1
        w2_c = c2
        if i==3:
            plt.axis([0,1,0,1])
            x = np.arange(0, 1, 0.1)
            if w == 0:
                w = 0.1
            y = x1+x2-x/w0
            plt.plot(x, y)
            plt.show()

        i = (i + 1) % 4
