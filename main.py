from Network import Network
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Menu:
    def __init__(self):
        self.network = Network()
        self.network_exist = False

    def read_int(self, message: str):
        try:
            i = int(input(message))
            return i
        except ValueError:
            print("Must be an int")
            return self.read_int(message)

    def read_float(self, message: str):
        try:
            i = float(input(message))
            return i
        except ValueError:
            print("Must be a float")
            return self.read_float(message)

    @staticmethod
    def read_data():
        print("Loading mnist dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # flattening matrix
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_test = (x_test.astype(np.float32) - 127.5) / 127.5
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_train = [[1 if d == i else 0 for i in range(10)] for d in y_train]
        y_test = np.array([[1 if d == i else 0 for i in range(10)] for d in y_test])
        # validation
        no_images = (10 * len(x_train)) // 100
        x_valid = np.array(x_train[len(x_train) - no_images:len(x_train)]).astype(np.float32)
        x_train = np.array(x_train[0:len(x_train) - no_images]).astype(np.float32)
        y_valid = np.array(y_train[len(y_train) - no_images:len(y_train)])
        y_train = np.array(y_train[0:len(y_train) - no_images])
        print("Dataset loaded!")
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def use_predetermined(self):
        self.network = Network()
        self.network.add_layer(784, 16)
        self.network.add_layer(16, 16)
        self.network.add_layer(16, 10)
        self.network.set_batches(28)
        self.network.set_learning_rate(0.001)
        self.network.set_epochs(3)

    def add_layer2network(self):
        if not self.network_exist:
            i = self.read_int("Input size = ")
            o = self.read_int("Output size = ")
            self.network_exist = True
            self.network.add_layer(i, o)
        else:
            o = self.read_int("Output size = ")
            self.network.add_layer(o, o)

    def set_hyper_parameters(self):
        b = self.read_int("Batch size: ")
        lr = self.read_float("Learning rate: ")
        e = self.read_int("Number of epochs: ")
        self.network.set_batches(b)
        self.network.set_learning_rate(lr)
        self.network.set_epochs(e)

    @staticmethod
    def main_options():
        print("1. Use predetermined\n"
              "2. Make your own network\n"
              "3. Train network\n"
              "4. Get results\n"
              "5. See example\n"
              "0. Exit\n")

    @staticmethod
    def network_options():
        print("1. Add layer to network\n"
              "2. Adjust hyperparameters\n"
              "0. Back\n")

    def network_menu(self):
        while True:
            self.network_options()
            choice = self.read_int("Your choice: ")
            if choice == 1:
                self.add_layer2network()
            elif choice == 2:
                self.set_hyper_parameters()
            elif choice == 0:
                print("Going back")
                break
            else:
                print("Choice not valid!")

    def main_menu(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.read_data()
        while True:
            self.main_options()
            choice = self.read_int("Your choice: ")
            if choice == 1:
                self.use_predetermined()
            elif choice == 2:
                self.network_menu()
            elif choice == 3:
                self.network.fit(x_train, y_train, x_valid, y_valid)
            elif choice == 4:
                self.network.evaluate(x_train, y_train, t="train")
                self.network.evaluate(x_valid, y_valid, t="valid")
                self.network.evaluate(x_test, y_test, t="test")
            elif choice == 5:
                keys = np.array(range(x_train.shape[0]))
                key = np.random.choice(keys)
                print(self.network.forward(x_train[key]))
                plt.imshow((x_train[key].reshape(28, 28)))
                plt.show()
            elif choice == 0:
                print("Leaving...")
                break
            else:
                print("Choice not valid!")


if __name__ == '__main__':
    menu = Menu()
    menu.main_menu()
