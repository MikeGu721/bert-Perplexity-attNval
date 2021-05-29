import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertForMaskedLM

text_formatter = lambda a: "[CLS]我的{}是{}[SEP]".format(a[0], a[1])

class Perplexity_Checker(object):
    def __init__(self, MODEL_PATH, MODEL_NAME, CACHE_DIR, device='cpu'):
        if MODEL_PATH:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            self.model = BertForMaskedLM.from_pretrained(MODEL_PATH)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
            self.model = BertForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.model.to(device)
        self.model.eval()
        self.DEVICE = device

    def add_device(self, DEVICE):
        self.DEVICE = DEVICE
        self.model.to(DEVICE)

    def sentence_preprocese(self, text):

        a,b = text
        temp_text = '"我的%s是%s。"' % (a, b)
        text = text_formatter(text)
        tokenized_text = np.array(self.tokenizer.tokenize(text))
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []
        for mask_word in [a, b]:
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            for masked_index in range(start_point + temp_text.index(mask_word) - 1,
                                      start_point + temp_text.index(mask_word) + len(mask_word) - 1):
                new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)

        segments_ids = np.tile(segments_ids, (2, 1))
        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def perplexity(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)

        a,b = text
        temp_text = '我的%s是%s。' % (a, b)

        tokens_tensor = torch.LongTensor(indexed_tokens)
        segments_tensors = torch.LongTensor(segments_ids)

        tokens_tensor = tokens_tensor.to(self.DEVICE)
        segments_tensors = segments_tensors.to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = torch.softmax(outputs[0], -1)
        total_perplexity = {}
        for i,masked_word in enumerate([a,b]):
            total_perplexity[[a,b][i]] = []
            for index in range(temp_text.index(masked_word),temp_text.index(masked_word)+len(masked_word)):
                total_perplexity[[a,b][i]].append(-np.log(predictions[i, index+1, real_indexs[index]].item()))
            total_perplexity[[a,b][i]] = np.array(total_perplexity[[a,b][i]]).sum()/len(masked_word)

        return total_perplexity


if __name__ == '__main__':
    import gzh

    # 模型名字
    MODEL_NAME = gzh.bert_name

    # 模型存放地址
    CACHE_DIR = gzh.cache_dir

    # 或者直接写模型存放地址
    MODEL_PATH = gzh.wwm_bert_path
    aa = Perplexity_Checker(MODEL_PATH, MODEL_NAME, CACHE_DIR, device='cuda')

    pairs = [['性别','男']]

    # 只MASK掉p和o
    for p,o in pairs:
        text = [p, o]
        try:
            p = aa.perplexity(text)
        except Exception as e:
            print(e,'——>',o)
        print(o + '：', p,np.array(list(p.values())).sum())
