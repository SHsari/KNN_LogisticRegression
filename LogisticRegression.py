import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import random
import numpy as np
np.set_printoptions(precision=3, suppress=True, edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "%.3g" % x))

class LoRe:
    def __init__(self, targetC):
        self.targetC = targetC
        (x_train, t_train), (x_test, t_test) = \
            load_mnist(flatten=True, normalize=True)

        self.batch_size = 1000   # 전체 Train 데이터 중 몇 개를 뽑아 학습 할 것인가
        self.test_size = 100     # 몇개를 Test 할 것인가. (Max = 10000)
        tmp_tr = np.full((self.batch_size, 1), 1)
        tmp_te = np.full((self.test_size,1), 1)

        batch_mask = random.sample(range(60000), self.batch_size)
        test_mask = random.sample(range(10000), self.test_size)
        self.w = np.full((785, 10), 0, float)  #bias로 인해 785

        self.x_batch = np.append(x_train[batch_mask], tmp_tr, axis=1)
        self.t_batch = t_train[batch_mask]
        self.x_test = np.append(x_test[test_mask], tmp_te, axis=1)
        self.t_test = t_test[test_mask]

        self.TF = np.where(self.t_batch == targetC, 1, 0)
        # print(self.TF.shape)
        # print(self.x_batch.shape)
        # print(self.t_batch.shape)
        # print(self.x_test.shape)
        # print(self.t_test.shape)
        # print('init end')


    def sigmoid(self, xw):
        xw = np.clip(xw, -500, 2**16)
        return (1 / (1 + np.exp(-xw)))
    """
    def cross_entropy_T(self, sig_xw):
        return -np.sum(np.log(sig_xw + 0.00001))/len(sig_xw)
    def cross_entropy_F(self, sig_xw):
        return -np.sum(np.log(1-sig_xw + 0.00001))/len(sig_xw)

    def gradient1(self, w, x_tr):
        h = 1/(2**32)
        dw = np.full(785, 0, float)
        for i in range(len(w)):
            w[i] += (-h)
            sig_xw = self.sigmoid(np.dot(x_tr, w))
            ipt = self.cross_entropy_T(sig_xw[self.TF]) + self.cross_entropy_F(sig_xw[~self.TF])
            w[i] += (2*h)
            sig_xw = self.sigmoid(np.dot(x_tr, w))
            fpt = self.cross_entropy_T(sig_xw[self.TF]) + self.cross_entropy_F(sig_xw[~self.TF])
            dw[i] = (fpt-ipt)/(2*h)
        return dw

    def cost1(self, w):
        np.dot(self.x_batch, w)
        return
    """


    def cross_entropy(self, sig_xw):
       return -np.mean(self.TF * np.log(sig_xw + 0.00001) + (1 - self.TF) * np.log(1 - sig_xw + 0.00001), axis=0)

    def cost(self, w):
        return self.cross_entropy(self.sigmoid(np.dot(self.x_batch, w)))

    def gradient(self, w):
        h = 1 / (2 ** 32)
        dw = np.full(785, 0, float)
        for i in range(len(w)):
            w[i] += (-h)
            ipt = self.cost(w)
            w[i] += (2 * h)
            fpt = self.cost(w)
            dw[i] = (fpt - ipt) / (2 * h)
        return dw

    def learn_single_target(self, lr=0.01, epoch=50):
        w = self.w[:, self.targetC]
        # print("learn_single_target: w.shape: ",w.shape)
        for j in range(epoch):
            dW = lr * self.gradient(w)
            w += dW * (-lr)
            #print((dW * lr)[:784].reshape(28,28))
            #print(dW.dtype)
            #print(w[:784].reshape(28,28))
            print("epoch: {0} |||  cost: {1}" .format(j, self.cost(w)))
        self.w[:, self.targetC] = w.T


    def test_single_target(self):
        w = self.w[:, self.targetC]
        prediction = self.sigmoid(np.dot(self.x_test, w))
        acc = 0
        for i in range(len(prediction)):
            print("class: {0}, chance: {1}".format(self.t_test[i], prediction[i]))
            if self.t_test[i] == self.targetC:
                if prediction[i] > 0.5:
                    acc += 1
            else:
                if prediction[i] < 0.5:
                    acc += 1
        acc = acc/len(self.x_test)
        print("accuracy: ", acc)