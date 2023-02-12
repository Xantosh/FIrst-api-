from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import *
from .reinforce import DQN,q_learner,gen_epsilon_greed_policy
from collections import deque
import torch
import torch.nn as nn
n_state = 2
n_action =3 
n_hidden = 50 
lr =0.01
dqn = DQN(n_state, n_action, n_hidden,lr)
memory = deque(maxlen=1000)
# updated replay size
rein_learn = q_learner(dqn,4, n_action)
# Create your views here.



class BookApiView(APIView):
    def post(self,request):
        allBooks = Book.objects.all().values()
        episode = request.data["episode"]
        episode = int(episode)
        rein_learn.setEpisode(episode)
        rein_learn.copy()
        return Response({"Message":"Done copying the parameters"})
    
class ActionTransform(APIView):
    def post(self,request):
        Superstate = int( request.data["Superstate"])
        action = int(request.data["action"])
        transFunction = torch.tensor([[0,0,3,0,6,0,0,0,7,0],[1,0,4,0,6,0,0,0,8,0],[2,0,5,0,6,0,0,0,9,0]])
        data = transFunction[action,Superstate]
        return Response({"transformed_action":data})

class StateApiView(APIView):
    def post(self,request):
        #Book.objects.create(id=request.data["id"], title=request.data["title"],author=request.data["author"])
        state = (int(request.data['npc']),int(request.data['pc']))
        
        action = rein_learn.action(state)
        
         
#        book = Book.objects.all().filter(id=request.data["id"]).values()
        return Response({"Message":"New action", "action":action})
    

class Train(APIView):
    def post(self, request):
        npc = request.data["npc"]
        pc = request.data["pc"]
        state = (npc,pc)
        
        action = request.data["action"]
        reward = request.data["reward"]
        next_npc= request.data["next_npc"]
        next_pc= request.data["next_pc"]
        next_state = (next_npc,next_pc)
        is_done = request.data["is_done"]
        rein_learn.process(memory, state, reward,next_state, action, is_done)
        rein_learn.change_epsilon()
        for layer in dqn.model.children():
            if isinstance(layer, nn.Linear):
                print(layer.state_dict()['weight'])
                print(layer.state_dict()['bias'])
        return Response({"Message":"Yorokobe shonen : kotomine "})










        
