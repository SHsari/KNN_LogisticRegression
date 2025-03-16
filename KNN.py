import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import random
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "%.3g" % x))


class KNN:
    def __init__(self, k_value):

        (self.x_train, self.t_train), (self.x_test, self.t_test) = \
            load_mnist(flatten=True, normalize=True)    #노멀라이즈 =TRUE
        self.k_value = k_value
        self.KNN_classes = []   # 각 Test Case마다 가장 가까운 K개 점들이 속한 class를 저장합니다. (여기선 꽃의 품종)
        self.KNN_distance = []  # 각 Test Case마다 가장 가까운 K개의 점들과의 거리값을 저장하는 2차원 배열입니다.
        self.Y_predict = []     # 예측한 Y값들이 들어갈 배열입니다.
        self.err =0             # 정확도 계산을 위한 에러 변수
        self.train_size = 1000  # 전체 train 데이터 중 몇 개를 학습용으로 사용할 것인가
        self.test_size = 100    # 전체 test 데이터 중 몇개에 적용할 것인가.
        self.smpl_train = random.sample(range(0, 60000), self.train_size)   #학습용 샘플을 랜덤하게 뽑는다.
        self.smpl_test = random.sample(range(0, 10000), self.test_size)     #테스트용 샘플을 랜덤하게 뽑는다.
        self.answ_test = self.t_test[self.smpl_test]    # 정답 어레이
        print("go")

    def img_show(img):
        pil_img = Image.fromarray(np.unit8(img))
        pil_img.show()

    def eucDist(self, arr1, arr2):  # Calculate Euclidian Distance
        dist = arr1 - arr2
        dist = dist**2
        return sum(dist)

    def classifier_1(self, ind_test):  # Get K Nearest Neighbors for one test Data
        KNN_distance = [9999999 for i in range(self.k_value)]
        KNN_classes = [-1 for i in range(self.k_value)]
        for ind_train in self.smpl_train:
            dist = self.eucDist(self.x_test[ind_test], self.x_train[ind_train])
            if dist <= max(KNN_distance):  # 만약 계산된 점간의 Distance가 KNN중 하나가 될 만큼 작다면
                for i in range(0, len(KNN_distance)):  # 리스트에 저장합니다.
                    if dist <= KNN_distance[i]:  # distance가 동일한 데이터에 대해서는 자세히 고려하지 않았습니다.
                        KNN_distance.insert(i, dist)  # 거리가 동일할 경우 그냥 나중에 계산된 값이 우선순위가 높습니다.
                        KNN_distance.pop()
                        KNN_classes.insert(i, self.t_train[ind_train])
                        KNN_classes.pop()
                        break
        return [KNN_classes, KNN_distance]

    def majority_vote(self, lst):
        # K Nearest Neightbors 중에 최빈값을 구합니다.
        # 웹에서 파이썬 최빈값 검색 하여 긁어왔습니다.
        # https://codingdojang.com/scode/612?langby=python#answer-filter-area
        # 위의 풀이들 중 가장 추천을 많이 받은 풀이를 사용했습니다.
        freq = {x: lst.count(x) for x in set(lst)}
        if len(freq) == len(lst):
            return None
        else:
            return [x for x in freq.keys() if freq[x] == max(freq.values())][0]

    def majority_vote_weighted(self, lst, dist):
        # Weighted KNN 을 위한 Weighted 최빈값을 구하는 알고리즘입니다.
        # 위의 Majority_vote 함수의 내용을 일부 참고했습니다.
        # weight += (n/distance)
        elements = list(set(lst))
        weights = [0 for i in range(len(elements))]
        for i in range(len(elements)):
            for j in range(len(lst)):
                if elements[i] == lst[j]:
                    weights[i] += 5 / dist[j]  # weight을 여기서 계산합니다.
        freq = {elements[x]: weights[x] for x in range(len(set(lst)))}
        return[x for x in freq.keys() if freq[x] == max(freq.values())][0]

    def get_KNN(self, weighted=0):
        for test_ind in self.smpl_test:
            KNNs_and_Dist = self.classifier_1(test_ind)
            self.KNN_distance.append(KNNs_and_Dist[1])
            self.KNN_classes.append(KNNs_and_Dist[0])
            if weighted == 0:
                self.Y_predict.append(self.majority_vote(KNNs_and_Dist[0]))
            elif weighted != 0:
                self.Y_predict.append(self.majority_vote_weighted(KNNs_and_Dist[0], KNNs_and_Dist[1]))
            else:
                print("value of the parameter 'weighted' has problem.")
                return 1
            # 아래는 결과 출력을 위한 과정입니다.
        for i in range(len(self.smpl_test)):
            print("[{0:>4}]:::mnist   KNNs: {1}    prediction: {2}    ans: {3}"
                  .format(self.smpl_test[i], self.KNN_classes[i], self.Y_predict[i], self.answ_test[i]), end='')
            if self.Y_predict[i] != self.answ_test[i]:
                print("  <= Wrong")
                self.err+=1
            else:
                print("")
        print("test size: {0}, error: {1}, acc: {2}".format(self.test_size, self.err, 1 - self.err/self.test_size))


"""

"""