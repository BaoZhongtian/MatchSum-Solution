import os
import time
import json
import tqdm
import torch
import numpy
import transformers
from model import BertSum


class MatchSumServer:
    def __init__(self, bertsum_path, matchsum_path, gpu_used='0', reveal_top_n=3, bertsum_select_number=7,
                 batch_size=2):
        # Model Initialization
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_used
        self.reveal_top_n = reveal_top_n
        self.batch_size = batch_size
        self.bertsum_select_number = bertsum_select_number

        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_sum_model = BertSum('bert-base-uncased')
        checkpoint = torch.load(bertsum_path)
        self.bert_sum_model.load_state_dict(checkpoint['state_dict'])
        print('BertSum Model Parameter Load Completed')
        self.bert_sum_model.eval()

        self.match_sum_model = torch.load(matchsum_path)
        self.match_sum_model.eval()
        self.match_sum_model.cuda()
        print('MatchSum Model Parameter Load Completed')

        ########################################
        # Fold
        self.request_path = 'Request/'
        self.result_path = 'Result/'
        if not os.path.exists(self.request_path): os.makedirs(self.request_path)
        if not os.path.exists(self.result_path): os.makedirs(self.result_path)

    def summarize_bertsum(self, text):
        bert_sum_tokens, bert_sum_position = [101], []
        for index in range(len(text)):
            if text[index] == '': continue
            bert_sum_position.append(len(bert_sum_tokens))
            bert_sum_tokens.extend(
                self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text[index])))
            if len(bert_sum_tokens) > 512: break
        if len(bert_sum_tokens) >= 512:
            bert_sum_tokens = bert_sum_tokens[0:512]
            bert_sum_tokens[511] = 102
        else:
            bert_sum_tokens.append(102)
        bert_sum_tokens = numpy.array(bert_sum_tokens)[numpy.newaxis, :]
        print('BertSum Tokenizer Complete, Shape =', numpy.shape(bert_sum_tokens))

        bert_sum_predict = self.bert_sum_model(torch.LongTensor(bert_sum_tokens), [bert_sum_position]).softmax(dim=-1)
        bert_sum_predict = bert_sum_predict.detach().cpu().numpy()[:, 1]
        selected_sentence_id = []
        for index in range(self.bertsum_select_number):
            selected_sentence_id.append(numpy.argmax(bert_sum_predict))
            bert_sum_predict[numpy.argmax(bert_sum_predict)] = -1
        selected_sentence_id = sorted(selected_sentence_id)
        print('Selected Sentence ID =', selected_sentence_id)
        return selected_sentence_id

    def summarize_matchsum(self, text, candidate):
        total_text_tokenized = self.bert_tokenizer.encode_plus(
            ' '.join([_[0:-1] for _ in text]), max_length=512, pad_to_max_length=True, return_tensors='pt')
        total_text_input_ids = total_text_tokenized['input_ids']
        candidate_tokenized = self.bert_tokenizer.batch_encode_plus(
            candidate, max_length=512, pad_to_max_length=True, return_tensors='pt')
        candidate_input_ids = candidate_tokenized['input_ids']

        total_text_input_ids = total_text_input_ids.cuda()
        total_score = []
        for batch_index in tqdm.trange(0, candidate_input_ids.size()[0] - 1, self.batch_size):
            score = \
                self.match_sum_model(total_text_input_ids,
                                     candidate_input_ids[batch_index:batch_index + self.batch_size].unsqueeze(0).cuda(),
                                     total_text_input_ids)['score'].squeeze()
            score = score.detach().cpu().numpy()
            total_score.extend(score)

        return total_score

    def summarize_loop(self):
        while True:
            if len(os.listdir(self.request_path)) == 0:
                time.sleep(1)
                continue
            for filename in os.listdir(self.request_path):
                if os.path.isfile(os.path.join(self.request_path, filename)): break

            ######################################
            # Ignore Too Short Request
            try:
                with open(os.path.join(self.request_path, filename), 'r', encoding='UTF-8') as file:
                    raw_data = file.readlines()
                    raw_data = [_.replace('\n', '').lower() for _ in raw_data]
            except:
                continue

            if len(raw_data) < 5:
                total_result = []
                total_result.append(
                    {'Text': "The Text is TOO SHORT or NOT divided by line, please check the input.", 'Score': -9999})
                json.dump(total_result, open(self.result_path + filename, 'w'))
                os.remove(self.request_path + filename)
                continue

            try:
                os.remove(self.request_path + filename)
            except:
                continue

            ######################################
            # BertSum and its candidates generation
            bertsum_selected = self.summarize_bertsum(text=raw_data)
            candidate_sentence = []
            for indexX in range(len(bertsum_selected)):
                for indexY in range(indexX + 1, len(bertsum_selected)):
                    candidate_sentence.append(
                        ' '.join([raw_data[bertsum_selected[indexX]],
                                  raw_data[bertsum_selected[indexY]]]).replace('\n', ''))
                    for indexZ in range(indexY + 1, len(bertsum_selected)):
                        candidate_sentence.append(
                            ' '.join([raw_data[bertsum_selected[indexX]], raw_data[bertsum_selected[indexY]],
                                      raw_data[bertsum_selected[indexZ]]]).replace('\n', ''))

            #######################################
            matchsum_score = self.summarize_matchsum(text=raw_data, candidate=candidate_sentence)

            packaged_result = {'BertSumSelect': [int(_) for _ in bertsum_selected],
                               'Scores': [float(_) for _ in matchsum_score]}

            for top_n_index in range(self.reveal_top_n):
                top_position = numpy.argmax(matchsum_score)
                packaged_result['Top-%d' % (top_n_index + 1)] = candidate_sentence[int(top_position)]
                matchsum_score[int(top_position)] = -9999

            json.dump(packaged_result, open(self.result_path + filename, 'w'))
            time.sleep(1)


if __name__ == '__main__':
    server = MatchSumServer(gpu_used='2', bertsum_path='BertSum-Parameter.pkl',
                            matchsum_path='MatchSum_cnndm_bert.ckpt', batch_size=10)
    print('Start Waiting for Document:')
    server.summarize_loop()
