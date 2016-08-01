class Learner:
    """Main class to run the user interface and manage the netweork"""
    def __init__(self, selected_set=1, selected_network=1, selected_configuration=1):
        """
        Init the first values
        calls the networks runner
        """
        self.network = selected_network
        self.set = selected_set
        self.config = selected_configuration
        self.set_set()
        self.set_configuration()
        self.run_network()

    def run_network(self):
        """Runs the selected network"""
        network = {
            1: self.init_cnn,
            2: self.init_alex_net
        }
        network_func = network.get(self.network, lambda: "nothing")
        network_func()

    def set_set(self):
        """sets the selected set"""
        selected_set = {
            1: self.msint,
            2: self.cifar_10,
            3: self.cyst,
            4: self.not_mnist
        }
        set_func = selected_set.get(self.set, lambda: "network")
        set_func()

    def msint(self):
        """init mnist"""
        pass

    def cifar_10(self):
        """init cifar 10"""
        pass

    def cyst(self):
        """init cyst"""
        pass

    def not_mnist(self):
        """init not mnist"""
        pass

    def set_configuration(self):
        """sets the selected configuration"""
        pass

    def init_cnn(self):
        """init CNN network"""
        from cnn import Cnn
        cnn = Cnn()

    def init_alex_net(self):
        """init AlexNet network"""
        from myalexnet import AlexNet
        an = AlexNet()