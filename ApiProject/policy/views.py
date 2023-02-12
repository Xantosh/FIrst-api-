from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import *
from .reinforce import PolicyNetwork,ValueNetwork,Policy_learner
from collections import deque


n_state  =2
n_action = 3
n_hidden_p = 64
lr_p = 0.003
policy_net = PolicyNetwork(n_state, n_action, n_hidden_p , lr_p)

n_hidden_v = 64
lr_v = 0.003
value_net = ValueNetwork(n_state, n_hidden_v, lr_v)
gamma=0.9
policy_learn = Policy_learner(policy_net,value_net,gamma)

class BookApiView(APIView):
    def get(self, request):
        policy_learn.value_initilize()
        return Response({"Message":"Done with initialization"})
    def post(self, request):
        state = (int(request.data['npc']),int(request.data['pc']))
        action = policy_learn.calc_action(state)

        return Response({"Message":"New action","action":action})





class Train(APIView):
    def post(self, request):
        reward = request.data["reward"]
        policy_learn.get_reward(int(reward))
        return Response({"Message":"done adding reward"})

    def get(self, request):
       # policy_learn.calc_action((0,0))
       # policy_learn.get_reward(5)
        print("policy states",policy_learn.states)

        policy_learn.update_network()
        return Response({"Message":"Updated the neural network"})


