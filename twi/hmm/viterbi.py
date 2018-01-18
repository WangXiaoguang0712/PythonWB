import numpy as np
import copy
__author__ = 'T'


# start_probability = {'Healthy': 0.6, 'Fever': 0.4}
# transition_probability = {'Healthy': {'Healthy': 0.7, 'Fever': 0.3}, 'Fever': {'Healthy': 0.4, 'Fever': 0.6}}
# emission_probability = {'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
#                       'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}}




def viterbit(obs, states, s_pro, t_pro, e_pro):
    path = { s:[] for s in states} # init path: path[s] represents the path ends with s
    curr_pro = {}
    for s in states:
        curr_pro[s] = s_pro[s] * e_pro[s][obs[0]]
    for i in xrange(1, len(obs)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in states:
            max_pro, last_sta = max(((last_pro[last_state] * t_pro[last_state][curr_state] * e_pro[curr_state][obs[i]],
                                      last_state) for last_state in states))
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)

    # find the final largest probability
    max_pro = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
    # print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path


def viterbi_v1(states, obs, init_pro, tran_pro, emi_pro):
    path = {s:[] for s in states}
    print path
    # first step
    pre_pro = {}
    cur_pro = {}
    for s in states:
        cur_pro[s] = init_pro[s] * emi_pro[s][obs[0]]
    # second to the last
    for i in range(1, len(obs)):
        pre_pro = cur_pro
        cur_pro = {}
        for cur_state in states:
            #for pre_state in states:
            max_pro,pre_state =  max(((pre_pro[pre_state] * tran_pro[pre_state][cur_state] * emi_pro[cur_state][obs[i]], pre_state)  for pre_state in states), key=lambda x:x[0])
            path[cur_state].append(pre_state)
            path[cur_state].append(pre_pro[pre_state])
            cur_pro[cur_state] = max_pro
        print(path)
    for s in states:
        path[s].append(s)
        path[s].append(cur_pro[s])
    print(path)

def dict2matrix(data, dim1, dim2):
    mat = np.zeros((len(dim1), len(dim2)))
    for row in data:
        for col,val in data[row].items():
            mat[dim1.index(row)][dim2.index(col)] = val
    return mat

def viterbi(states, o, init_pro, tran_pro, emi_pro):
    path = {s:[] for s in states}
    prob = np.zeros(len(states))
    for i in range(len(o)):
        if i == 0:
            prob = init_pro * emi_pro[:, i]
        else:
            pre_prob = copy.deepcopy(prob)
            for s in range(len(states)):
                tmp = pre_prob * tran_pro[:, s] * emi_pro[s, o[i]]
                prob[s] = max(tmp)
                path[states[s]].append((states[list(tmp).index(prob[s])],str(pre_prob[list(tmp).index(prob[s])])))
            print(path)
    for s in range(len(states)):
        path[states[s]].append((states[s],str(prob[s])))
    ix = list(prob).index(max(prob))
    print(path[states[ix]])


if __name__ == '__main__':
    states = ('Healthy','Fever')
    obs = ['normal', 'cold', 'dizzy']
    start_probability = {'Healthy': 0.5, 'Fever': 0.5} # 0.6 0.4
    transition_probability = {'Healthy': {'Healthy': 0.6, 'Fever': 0.4}, 'Fever': {'Healthy': 0.4, 'Fever': 0.6}}
    emission_probability = {'Healthy': {'normal': 0.7, 'cold': 0.2, 'dizzy': 0.1},
                      'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}}
    o = [0,2,2,1,0,2,1]
    pai = [0.5, 0.5]
    ma = dict2matrix(transition_probability, states, states)
    mb = dict2matrix(emission_probability, states, obs)
    # viterbi_v1(states, obs, start_probability, transition_probability, emission_probability)
    viterbi(states, o, pai, ma, mb)