import numpy as np
import pandas as pd
from function import generate_data, get_weighted_sum, sigmoid, cross_entropy, update_weights, update_bias

bias = 0.5
l_rate = 0.1
epochs = 10
epoch_loss =[]

data, weight = generate_data(50,3)

def train_model(data, weight, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss =[]

    # data, weight = generate_data(4,3)
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weight, bias)
            # print(W_sum)
            prediction = sigmoid(w_sum)
            # print(prediction)
            loss = cross_entropy(target, prediction)
            # print(loss)
            individual_loss.append(loss)
        # print(data)
            # print("OLD VALUES")
            # print(weight, bias)
        #graaduent decent
            weight = update_weights(weight, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        # print("New VALUES")
        # print(weight, bias)
    average_loss = sum(individual_loss)/len(individual_loss)
    epoch_loss.append(average_loss)

    print("*********************************************")
    print("epoch", e)
    print(average_loss)

train_model(data, weight, bias, l_rate, epochs)


df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.saveFig("Training_Loss.pdf")

