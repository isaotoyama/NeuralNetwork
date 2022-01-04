import numpy as np
import pandas as pd
rg = np.random.default_rng()

def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    # print(features)
    weights = rg.random((1, n_values))[0]
    # print(weights)
    targets = np.random.choice([0,1], n_features)
    # print(targets)
    data = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    data["target"] = targets
    # print (data)
    return data, weights
    
def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias

def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

def cross_entropy(target, prediction): 
    # return -(target*np.log10(prediction)+(1-target)*np.log10(1-prediction))
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))
def update_weights(weight, l_rate, target, prediction, feature):
    new_weights = []
    for x,w in zip(feature, weight):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights

def update_bias(bias, l_rate, target, prediction):
   return  bias + l_rate*(target-prediction)