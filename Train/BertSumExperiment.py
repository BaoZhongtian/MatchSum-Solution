import os
import json
import numpy
import torch
from model import BertSum
from Train.tools import ProgressBar
from transformers import BertTokenizer


def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


gpus = "1"
batch_size = 8
learning_rate = 1E-4
epoch_number = 10
save_path = 'E:/ProjectData/BertSum-Result-All-WeightChange/'
encoder_name = 'bert-base-uncased'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    train_data = load_jsonl('../data/train_CNNDM_bert.jsonl')
    tokenizer = BertTokenizer.from_pretrained(encoder_name)

    model = BertSum(encoder_name)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step_counter = 0
    total_loss = 0.0
    pbar = ProgressBar(n_total=int(len(train_data) / batch_size) * epoch_number, desc='Training')
    for episode_index in range(epoch_number):
        for batch_index in range(0, len(train_data), batch_size):
            # if batch_index / 8 < 17520: continue
            treat_raw_data = [train_data[_]['text'] for _ in
                              range(batch_index, min(batch_index + batch_size, len(train_data)))]
            treat_label = [train_data[_]['label'] for _ in
                           range(batch_index, min(batch_index + batch_size, len(train_data)))]

            treat_raw_token, treat_raw_position = [], []
            for indexX in range(len(treat_raw_data)):
                current_token, current_position = [101], []
                for indexY in range(len(treat_raw_data[indexX])):
                    current_position.append(len(current_token))
                    current_token.extend(
                        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(treat_raw_data[indexX][indexY])))
                    if len(current_token) > 512: break

                if len(current_token) >= 512:
                    current_token = current_token[0:512]
                    current_token[511] = 102
                else:
                    current_token.append(102)

                assert len(current_token) == len(train_data[batch_index + indexX]['text_id'])
                if len(current_position) < len(treat_label[indexX]):
                    treat_label[indexX] = treat_label[indexX][0:len(current_position)]
                assert len(current_position) >= len(treat_label[indexX])
                current_position = current_position[0:len(treat_label[indexX])]

                treat_raw_token.append(current_token)
                treat_raw_position.append(current_position)

            batch_data, batch_position, batch_label = [], treat_raw_position, []
            for _ in treat_label: batch_label.extend(_)
            batch_label = torch.LongTensor(batch_label).cuda()

            for sample in treat_raw_token:
                batch_data.append(numpy.concatenate([sample, numpy.zeros(512 - len(sample))]))
            batch_data = torch.LongTensor(batch_data).cuda()

            loss = model(batch_data, batch_position, batch_label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            model.zero_grad()

            pbar(episode_index * int(len(train_data) / batch_size) + int(batch_index / 8),
                 {'loss': loss.item(), 'count': step_counter})

            step_counter += 1
            if step_counter % 1000 == 999:
                print('\nLoss =', total_loss)
                total_loss = 0.0

                torch.save(
                    {'epoch': episode_index, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    os.path.join(save_path, '%08d-Parameter.pkl' % step_counter))
