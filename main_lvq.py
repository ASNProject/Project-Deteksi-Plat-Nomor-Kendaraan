import pandas as pd
import numpy as np
from math import *
from sklearn.preprocessing import MinMaxScaler
import time

def lvq_fit(train, target, learn_rate, b, epsilon, m, max_epoch, data_per_class):
    if (data_per_class == 1):
      unique_label, index_unique = np.unique(target, return_index=True)  
    elif (data_per_class == 2):
      unique_label = [] 
      index_unique = []
      unik=0
      for x in target: 
          if x not in unique_label: 
            for jumlah in range(2):
                unique_label.append(x) 
            index_unique.extend((unik, unik+1)) 
          unik += 1
    else:
      raise ValueError

  
    weight = train[index_unique].astype(np.float64)
    print("Bobot Awal Proses Training \n", weight)
    
    train = np.array([e for i, e in enumerate(zip(train, target)) if i not in index_unique])
    train, target = train[:, 0], train[:, 1]
    epoch = 0

    while epoch < max_epoch:
        for i, x in enumerate(train):
            distance = [sqrt(sum((w - x) ** 2)) for w in weight]
            winner, runner_up = np.argsort(distance)[:2]
            # print(winner+1, runner_up+1, target[i])
            winner_same_target = unique_label[winner] == target[i]
            runner_same_target = unique_label[runner_up] == target[i]
            if (min(distance[winner]/distance[runner_up], distance[runner_up]/distance[winner]) > (1 - epsilon)*(1 + epsilon) and (winner_same_target or runner_same_target)):
              # print('ubah')
              if (winner_same_target and runner_same_target):
                # print('sama')
                beta = m * learn_rate
                weight[winner] += 1 * beta * (x - weight[winner])
                weight[runner_up] += 1 * beta * (x - weight[runner_up])
              else:
                # print('beda')
                sign = 1 if winner_same_target else -1
                weight[winner] += sign * learn_rate * (x - weight[winner])
                weight[runner_up] -= sign * learn_rate * (x - weight[runner_up])

        # print(f'epoch = {epoch} \n', weight)
        learn_rate *= b
        epoch += 1

    return weight, unique_label

def lvq_predict(x_pred, weight, label):
    return np.asarray([label[np.argmin([sqrt(sum((w - x) ** 2)) for w in weight])] for x in x_pred])

def hitungAkurasi(predict_class, real_class):
    temp = sum(1 for predict, real in zip(predict_class, real_class) if predict == real)
    accuration = (temp / len(predict_class)) * 100
    return accuration

def main():
    url = 'https://raw.githubusercontent.com/xabela/lvq3/master/Data-Latih.csv'
    df= pd.read_csv(url)
    # print("Data Train \n", df)

    x = df.values[:, :-1]
    y = df.values[:, -1]

    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(x)
    # print("Normalisasi Data Train", data_train)
    
    train_weight, labels = lvq_fit(data_train, y, learn_rate=.2, b=.5, epsilon=.2, m=.3, max_epoch=10, data_per_class=2)
    print("Hasil Training \n", train_weight, labels)

    url2 = 'https://raw.githubusercontent.com/xabela/lvq3/master/Data-Uji.csv'
    df= pd.read_csv(url2)
    # print("Data Uji \n", df)

    x = df.values[:, :-1]
    y = df.values[:, -1]

    data_test = scaler.transform(x)
    # print("Normalisasi Data Test \n", data_test)

    y_pred = lvq_predict(data_test, train_weight, labels)
 
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Runtime : %s sec" % (time.time() - start_time)) 
