from datasets import load_dataset
import torch
import os

import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from torch.utils.data import Dataset
from pathlib import Path
class EmotionAnalyzer():

    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased'
        )
        self.Mapper = {
            0:"sadness",
            1: "joy",
            2:"love",
            3:"anger",
            4:"fear",
            5: "surprise"
        }

    def GetEmotion(self, sentence):
        DEVICE = self.DEVICE
        tokenizer = self.tokenizer
        sentence_encodings = tokenizer([sentence], truncation = True, padding= True)
        pth = os.path.dirname(__file__)
        model = DistilBertForSequenceClassification.from_pretrained(
           os.path.join(pth,'emotionsx\\'),
            num_labels = 6
        )
        outputs = model(torch.tensor(sentence_encodings['input_ids']),attention_mask =torch.tensor( sentence_encodings['attention_mask']))
        logits = outputs['logits']
        x = torch.argmax(logits,1)
        return self.Mapper[x.item()]
    


EA = EmotionAnalyzer()
EA.GetEmotion("i really like you")
# model = DistilBertForSequenceClassification.from_pretrained(
#            './emotionsx/',
#             num_labels = 6)