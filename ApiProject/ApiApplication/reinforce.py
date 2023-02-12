import torch
from collections import deque
import random
import copy
from torch.autograd import Variable
import numpy as np

class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state,n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr)

        self.model_target = copy.deepcopy(self.model)

    def target_predict(self, s):
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())
    
    def update(self, s,y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))


                
    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory , replay_size)
            states =[]
            td_targets =[]
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma* torch.max(q_values_next).item()

                td_targets.append(q_values)

            states = np.array(states)
            self.update(states,td_targets)

       

def gen_epsilon_greed_policy(estimator,epsilon, n_action ):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0,n_action-1)

        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()

    return policy_function






class q_learner():
    def __init__(self, estimator ,  replay_size,n_action, target_update=10, gamma=1.0, epsilon=0.1 , epsilon_decay=0.99):
        self.estimator = estimator 
 #       self.n_episide = n_episode
        self.replay_size = replay_size
        self.target_update = target_update
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episode = 0
        self.policy = gen_epsilon_greed_policy(self.estimator,self.epsilon,  n_action )

    def setEpisode(self, episode):
        self.episode = episode

    def copy(self):
        if self.episode % self.target_update :
            self.estimator.copy_target()
                
    def action(self,state):
        action = self.policy(state)
        return action

    def process(self,memory,state,reward,next_state,action,is_done):
        memory.append((state, action, next_state, reward, is_done))

        self.estimator.replay(memory, self.replay_size, self.gamma)

        state = next_state
        return state

    def change_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

# call these process in the order according to the algorithm in ipynb
