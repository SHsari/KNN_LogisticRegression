import matplotlib.pyplot as plt
import numpy as np
import pickle

class MKplot:
    def __init__(self):
        size_data = 785
        cls_num = 10
        self.w = np.full((size_data, cls_num), 0, float)
        self.lr = 0
        self.epoch = 0
        self.batch_size =0
        self.cost = np.full((self.batch_size, 10), -1, float)

    def makeplt(self, filename):
        try:
            with open(filename, 'rb') as file:
                self.w = pickle.load(file)
                self.lr = pickle.load(file)
                self.epoch = pickle.load(file)
                self.batch_size = pickle.load(file)
                self.cost = pickle.load(file)
        except:
            print('load weight error: {0}'.format(filename))
        cost = self.cost.T
        x = range(self.epoch)
        plt.figure()
        for i in range(len(cost)):
            plt.plot(x, cost[i], label='{}'.format(i))
            plt.xlabel('epoch')
            plt.ylabel('cost')
            plt.title('lr = {}'.format(self.lr))
            plt.legend()

        plt.show()

