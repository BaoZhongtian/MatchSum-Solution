import torch
from torch import nn
from torch.nn import init

from transformers import BertModel, RobertaModel


class MatchSum(nn.Module):
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()

        self.hidden_size = hidden_size
        self.candidate_num = candidate_num

        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text_id, candidate_id, summary_id):
        batch_size = text_id.size(0)

        pad_id = 0  # for BERT
        if text_id[0][0] == 0:
            pad_id = 1  # for RoBERTa

        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]  # last layer
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size)  # [batch_size, hidden_size]

        # get summary embedding
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0]  # last layer
        summary_emb = out[:, 0, :]
        assert summary_emb.size() == (batch_size, self.hidden_size)  # [batch_size, hidden_size]

        # get summary score
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num,
                                          self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)

        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)  # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}


# Special Note: This model is not existed in https://github.com/maszhongming/MatchSum
# I have added it by myself understand of BertSum paper.
class BertSum(torch.nn.Module):
    def __init__(self, model_name):
        assert model_name in ['bert-base-uncased', 'bert-large-uncased']
        super(BertSum, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        if model_name == 'bert-base-uncased':
            self.predict_layer = torch.nn.Linear(in_features=768, out_features=2)
        if model_name == 'bert-large-uncased':
            self.predict_layer = torch.nn.Linear(in_features=1024, out_features=2)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, text_id, text_position, label=None):
        pad_id = 0  # for BERT
        if text_id[0][0] == 0:
            pad_id = 1  # for RoBERTa

        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]  # last layer

        assert len(text_position) == out.size()[0]
        start_word_result = []
        for indexX in range(len(text_position)):
            for indexY in range(len(text_position[indexX])):
                start_word_result.append(out[indexX][text_position[indexX][indexY]].unsqueeze(0))
        start_word_result = torch.cat(start_word_result, dim=0)

        predict = self.predict_layer(start_word_result)
        if label is None: return predict

        assert predict.size()[0] == label.size()[0]
        loss = self.loss(input=predict, target=label)
        return loss
