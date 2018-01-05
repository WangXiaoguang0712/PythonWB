#coding:utf-8
import numpy as np
import pickle

states = ('Rainy', 'Sunny')

observations = ('walk', 'shop', 'clean')

start_probability = {'Rainy': 0.6, 'Sunny': 0.4}

transition_probability = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
}

emission_probability = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}

if __name__ == '__main__':
    predictions = []
    new_probability = {}
    for day, action in enumerate(observations):
        new_emission_probability = {}
        if day == 0:
            for weather, prob in start_probability.items():
                new_probability[weather] = prob * emission_probability[weather][action]
            print(new_probability)
            predictions.append(
                max(new_probability.items(), key=lambda x: x[1])[0]
            )
        else:
            for weather, prob in new_probability.items():
                new_probability[weather] = transition_probability[predictions[-1]][weather] * emission_probability[weather][action]
            print(new_probability)
            predictions.append(
                max(new_probability.items(), key=lambda x: x[1])[0]
            )
    print(predictions)
