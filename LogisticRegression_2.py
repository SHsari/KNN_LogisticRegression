import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import random
import pickle
import numpy as np
np.set_printoptions(precision=3, suppress=True, edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "%.3g" % x))


class LoRe2:
    def __init__(self, batch_size, test_size):
        # 하나의 클래스 안에서는 학습과 테스트 데이터 셋이 고정입니다.

        (x_train, t_train), (x_test, t_test) = \
            load_mnist(flatten=True, normalize=True)    #normalize = True
        self.clsnum =10
        self.batch_size = batch_size  # 전체 Train 데이터 중 몇 개를 뽑아 학습 할 것인가
        self.test_size = test_size

        batch_mask = random.sample(range(len(x_train)), self.batch_size)
        tmp_tr = np.full((self.batch_size, 1), 1)  # bias항을 만들기 위해 tmp_tr을 정의합니다.
        self.x_batch = np.append(x_train[batch_mask], tmp_tr, axis=1)  # x_batch에 bias(상수항=1) 추가
        self.t_batch = t_train[batch_mask]
        self.TF = np.full((self.batch_size, self.clsnum), 0)  # index mask를 만들기
        self.TF[np.arange(self.batch_size), self.t_batch] = 1  # index mask 성공! python indexing에 대한 개념을 배웠습니다.

        test_mask = random.sample(range(len(x_test)), test_size)    #test sample을 뽑기 위한 마스크 입니다.
        tmp_te = np.full((test_size, 1), 1)
        self.x_test = np.append(x_test[test_mask], tmp_te, axis=1)  #마지막 785번째 열을 만들어 1로 패딩
        self.t_test = t_test[test_mask]

        # 특정 lr 또는 epoch에서의 학습 결과(weight 과 epoch 마다의 cost)를 저장하기 위하 아래 t_로 시작하는 아래 변수들을 만들었습니다.
        self.t_weight = np.full((785, self.clsnum), 0, float)
        self.t_lr = 0
        self.t_epoch = 0
        self.t_batch_size = 0
        self.t_cost = np.full((self.t_epoch, self.clsnum), -1)

    def sigmoid(self, xw):
        xw = np.clip(xw, -500, 2**16)           #np.exp에서 overflow가 자꾸 발생하여 웹에서 찾은 해법입니다.
        return (1 / (1 + np.exp(-xw)))

    def ReLU(self, xw):
        return np.maximum(0, xw)                #시그모이드 대체로 해봤는데 잘 작동하지 않습니다.

    # 아래 cross_entropy 함수는 batch_size * 10(클래스 갯수) 크기의 행렬을 인풋으로 받으면
    # 각 행과 열에 대한 계산 결과값을 axis=0(각 열에 대해(?)) 에 대해 평균을 내어 길이 10(클래스 갯수)의 벡터를 반환한다.
    def cross_entropy(self, sig_xw):
        return -np.mean(self.TF * np.log(sig_xw + 0.00001) + (1 - self.TF) * np.log(1 - sig_xw + 0.00001), axis=0)

    def cost(self, x, w):   #매우 간단하게 구현되는 것을 보면 놀랍습니다.
        return self.cross_entropy(self.sigmoid(np.dot(x, w)))

    def gradient(self, x, w):       #gradient를 for 문 없이 행렬계산으로 한번에 할 수 없을까 하는 생각이 좀 듭니다.
        # 각 w에 대한 편미분값 * (-1) 을 반환
        h = 1 / (2 ** 16)
        dW = np.full((785, self.clsnum), 0, float)
        for i in range(len(w)):
            w[i] -= h
            ipt = self.cost(x, w)
            w[i] += 2 * h
            fpt = self.cost(x, w)
            dW[i] = (ipt - fpt) / (2 * h)  # 추후 dW 값에 -를 곱할 필요 없습니다.
        return dW

    def learn_Multiclass(self, filename: str, learning_rate=0.01, epoch=20):
        lr = learning_rate
        w = np.full((785, self.clsnum), 0, float)  # weight값을 담은 행렬의 행 갯수는 input 갯수 784 와 bias행을 더해 785 입니다.
        cost = np.full((epoch, self.clsnum), -1, float)
        for j in range(epoch):
            dW = self.gradient(self.x_batch, w)
            w += (lr * dW)                # 각 w에 대한 미분값 * (-1)을 반환
            cost[j] = self.cost(self.x_batch, w)
            print("epoch {0}: cost: {1}".format(j, cost[j]))
        print("learning_rate: {0},   |epoch: {1}".format(lr, epoch))
        self.save_weight(filename, w, lr, epoch, cost)

    def save_weight(self, file_name, weight, learning_rate, epoch, cost):       # 학습결과 저장
        try:
            with open(file_name, 'wb') as file:
                pickle.dump(weight, file)
                pickle.dump(learning_rate, file)
                pickle.dump(epoch, file)
                pickle.dump(self.batch_size, file)
                pickle.dump(cost, file)
        except:
            print('save weight error: {0}'.format(file_name))

    def test_Multiclass(self, file_name=''):    # 저장된 학습 결과를 바탕으로 테스트 하는 함수
        try:                                    # 좀 급하게 짜서 정리되지 않은 부분이 있는 것 같습니다.
            self.load_weight(file_name)
        except:
            print("filename \'{}\' not correct", file_name)
        prediction = self.sigmoid(np.dot(self.x_test, self.t_weight))   # 데이터가 각 클래스일 확률을 계산합니다..
        t_pred = np.argmax(prediction, axis=1)
        for i in range(len(prediction)):
            str = "\n {0:4d}. " .format(i)
            ind = np.where(prediction[i] > 0.4)[0]  #확률 0.2 이상의 클래스를 보여줍니다.
            for j in ind:
                str += " p{0}: {1:5.2f}% | ".format(j, prediction[i][j] * 100)
            print("%-60s" % str, end=' ')
            print("ans {0}, p_Max {1}, {2:5.2f}"\
                  .format(self.t_test[i], t_pred[i], prediction[i][t_pred[i]]), end='')
        accu = len(np.where(t_pred == self.t_test)[0])
        print("\n     accuracy: {0:5.2f}% , {1}/{2}".format(accu/self.test_size*100, accu, self.test_size))
        print("learning rate: {0}, epoch: {1}".format(self.t_lr, self.t_epoch))

    def load_weight(self, file_name):       #저장된 학습 결과 파일을 불러옴.
        try:
            with open(file_name, 'rb') as file:
                self.t_weight = pickle.load(file)
                self.t_lr = pickle.load(file)
                self.t_epoch = pickle.load(file)
                self.t_batch_size = pickle.load(file)
                self.t_cost = pickle.load(file)
        except:
            print('load weight error: {0}'.format(file_name))
