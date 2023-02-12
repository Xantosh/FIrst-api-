import torch
from collections import deque
import random
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

class PolicyNetwork():
    def __init__(self,n_state,n_action,n_hidden=50,lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_action),
            nn.Softmax()
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self,s):
        return self.model(torch.Tensor(s))

    def update(self, advantags, log_probs):
        policy_gradient =[]
        for log_prob , Gt in zip(log_probs, advantags):
            policy_gradient.append(-log_prob* Gt)
        
        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, s):
        probs = self.predict(s)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action , log_prob


class ValueNetwork():
    def __init__(self, n_state, n_hidden=50, lr = 0.05):
        self.criterion = nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,1)

        )
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr)

    
    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred,Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))


class Policy_learner():
    def __init__(self,estimator_policy,estimator_value, gamma =1 ):
        self.estimator_policy = estimator_policy
        self.estimator_value = estimator_value

        self.gamma = gamma
        self.log_probs = []
        self.states =[]
        self.rewards =[]
        
        
    def value_initilize(self):
        self.log_probs = []
        self.states =[]
        self.rewards =[]

    
    def calc_action(self,state):
        self.states.append(state)
        action, log_prob = self.estimator_policy.get_action(state)
        self.log_probs.append(log_prob)
        return action

    def get_reward(self, reward):
        self.rewards.append(reward)        


    def update_network(self):
        Gt =0 
        pw =0 
        returns = []
        for t in range(len(self.states) - 1 , -1,-1):
            print(self.rewards)
            Gt += self.gamma** pw * self.rewards[t]
            pw += 1
            returns.append(Gt)
            
        returns = returns[::-1]
        returns = torch.tensor(returns)
        baseline_values = self.estimator_value.predict(self.states)

        advantages = returns - baseline_values
        self.estimator_value.update(self.states, returns)
        self.estimator_policy.update(advantages, self.log_probs)


    


