import sys

from learner import Learner

if __name__ == "__main__":
    # if got arguments
    if len(sys.argv) == 4:
        set_num = int(sys.argv[1])
        network = int(sys.argv[2])
        conf = int(sys.argv[3])
    else:
        try:
            set_num = int(input("Enter Set Number: "))
            network = int(input("Enter Network Number: "))
            conf = int(input("Enter the Selected Configuration: "))
        except:
            print "Wrong Arguments"
            raise SystemExit

    # Check for Exceptions
    if set_num < 1 or set_num > 4:
        print "Wrong Set Value"
        raise SystemExit
    if network < 0 or network > 2:
        print "Wrong Network Value"
        raise SystemExit
    if conf < 0 or conf > 2:
        print "Wrong Configuration Value"
        raise SystemExit

    # prettify the print
    set_print = {
        1: "MNIST",
        2: "CIFAR-10",
        3: "Cyst",
        4: "Not_MNIST"
    }
    network_print = {
        1: "CNN",
        2: "AlexNet"
    }
    # conf_print = {
    #
    # }
    print "Selected {0} Network with {1} Set".format(network_print[network], set_print[set_num])
    print "Crafting your selected settings..."
    # start running the learner
    learner = Learner(set_num, network, conf)
