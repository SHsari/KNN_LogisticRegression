from KNN import *
from LogisticRegression import *
from LogisticRegression_2 import *
from MKplot import *


# # 1. Single Target Learning & test
# sLR = LoRe(targetC= 1)    #target class를 지정해주세요
# sLR.learn_single_target(lr=0.4, epoch=50)
# sLR.test_single_target()


# # 2-1. Multiclass Logistic Regression (learning)
# # 아래 10개의 learning rate 에 대해 학습하고 결과를 저장합니다.
#
# pl = LoRe2(batch_size=1000, test_size=500)
# lr = [2, 1.5, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
# epoch = 200
#
# for i in range(10):
#     str = "Auto_test{}.p".format(i)
#     print(str)
#     pl.learn_Multiclass(str, lr[i], epoch)


# # 2-2. Multiclass LR Test only w/pre-calculated Weights
# # Learning_Rate는 위의 lr 리스트를 확인해주세요, epoch =200로 학습한 weight에 대한 테스트 입니다. 변수파일명은
# # 'Auto_test{}.p' 형식이며 중괄호에 0~9까지 번호를 입력하면 각 learning rate로 학습한 Weight을 test에 대입합니다.
# # 확률이 0.4 이상인 모든 클래스들을 보여줍니다.
# # 10개 클래스 중 확률이 가장 높은 클래스로 정답 여부를 판별했습니다. 우측에 확률이 가장 높은 클래스와 그 확률이 나옵니다.
#
pt = LoRe2(batch_size=1000, test_size=500)
pt.test_Multiclass("Auto_test4.p")


# # 3. 아래는 이미 계산된 값에 대해 plot을 보여줍니다.
# # epoch가 증가함에 따라 감소하는 cost를 볼 수 있습니다.
#
# mkplt0 = MKplot()
#
# for i in range(10):
#     str = "Auto_test{}.p".format(i)
#     mkplt0.makeplt(str)
# plt.show()


# # 4. KNN입니다.
# knn = KNN(k_value=9)
# knn.get_KNN(weighted=1)





