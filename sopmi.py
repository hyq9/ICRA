#!/usr/bin/env python3
# coding: utf-8
# File: so-pmi.py
# Date: 18-4-4

import jieba.posseg as pseg
import jieba
import math
import time

class SoPmi:
    def __init__(self):
        self.train_path = './data/train.txt'  # 训练数据路径
        self.candipos_path = 'data/candi_pos.txt'  # 积极候选词保存路径
        self.candineg_path = 'data/candi_neg.txt'  # 消极候选词保存路径
        self.sentiment_path = './data/sentiment_words.txt'  # 情感词库路径

    def seg_corpus(self, train_data, sentiment_path):
        sentiment_words = [line.strip().split('\t')[0] for line in open(sentiment_path, 'r', encoding='utf-8')]
        for word in sentiment_words:
            jieba.add_word(word)
        seg_data = []
        for line in open(train_data, 'r', encoding='utf-8'):
            line = line.strip()
            if line:
                words = [word.word for word in pseg.cut(line) if word.flag[0] not in ['u', 'w', 'x', 'p', 'q', 'm']]
                seg_data.append(words)
        return seg_data

    def collect_cowords(self, sentiment_path, seg_data):
        cowords_list = list()
        window_size = 5
        sentiment_words = [line.strip().split('\t')[0] for line in open(sentiment_path, 'r', encoding='utf-8')]
        for sent in seg_data:
            if set(sentiment_words).intersection(set(sent)):
                for index, word in enumerate(sent):
                    if index < window_size:
                        left = sent[:index]
                    else:
                        left = sent[index - window_size: index]
                    if index + window_size > len(sent):
                        right = sent[index + 1:]
                    else:
                        right = sent[index: index + window_size + 1]
                    context = left + right + [word]
                    if set(sentiment_words).intersection(set(context)):
                        for index_pre in range(0, len(context)):
                            if set([context[index_pre]]).intersection(sentiment_words):
                                for index_post in range(index_pre + 1, len(context)):
                                    cowords_list.append(context[index_pre] + '@' + context[index_post])
        return cowords_list

    def collect_candiwords(self, seg_data, cowords_list, sentiment_path):
        word_dict, all = self.collect_worddict(seg_data)
        co_dict, candi_words = self.collect_cowordsdict(cowords_list)
        pos_words, neg_words = self.collect_sentiwords(sentiment_path, word_dict)
        pmi_dict = self.compute_sopmi(candi_words, pos_words, neg_words, word_dict, co_dict, all)
        return pmi_dict

    def collect_worddict(self, seg_data):
        word_dict = dict()
        for line in seg_data:
            for word in line:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        all = sum(word_dict.values())
        return word_dict, all

    def collect_cowordsdict(self, cowords_list):
        co_dict = dict()
        candi_words = list()
        for co_words in cowords_list:
            candi_words.extend(co_words.split('@'))
            if co_words not in co_dict:
                co_dict[co_words] = 1
            else:
                co_dict[co_words] += 1
        return co_dict, candi_words

    def collect_sentiwords(self, sentiment_path, word_dict):
        pos_words = set([line.strip().split('\t')[0] for line in open(sentiment_path, 'r', encoding='utf-8') if line.strip().split('\t')[1] == 'pos']).intersection(set(word_dict.keys()))
        neg_words = set([line.strip().split('\t')[0] for line in open(sentiment_path, 'r', encoding='utf-8') if line.strip().split('\t')[1] == 'neg']).intersection(set(word_dict.keys()))
        return pos_words, neg_words

    def compute_sopmi(self, candi_words, pos_words, neg_words, word_dict, co_dict, all):
        pmi_dict = dict()
        for candi_word in set(candi_words):
            pos_sum = 0.0
            neg_sum = 0.0
            for pos_word in pos_words:
                pair = pos_word + '@' + candi_word
                if pair in co_dict:
                    p1 = word_dict[pos_word] / all
                    p2 = word_dict[candi_word] / all
                    p12 = co_dict[pair] / all
                    pos_sum += math.log2(p12 / (p1 * p2))

            for neg_word in neg_words:
                pair = neg_word + '@' + candi_word
                if pair in co_dict:
                    p1 = word_dict[neg_word] / all
                    p2 = word_dict[candi_word] / all
                    p12 = co_dict[pair] / all
                    neg_sum += math.log2(p12 / (p1 * p2))

            so_pmi = pos_sum - neg_sum
            pmi_dict[candi_word] = so_pmi
        return pmi_dict

    def save_candiwords(self, pmi_dict, candipos_path, candineg_path):
        pos_dict = {word: score for word, score in pmi_dict.items() if score > 0}
        neg_dict = {word: abs(score) for word, score in pmi_dict.items() if score < 0}

        with open(candipos_path, 'w', encoding='utf-8') as f_pos, open(candineg_path, 'w', encoding='utf-8') as f_neg:
            for word, score in sorted(pos_dict.items(), key=lambda item: item[1], reverse=True):
                f_pos.write(f"{word},{score},pos\n")
            for word, score in sorted(neg_dict.items(), key=lambda item: item[1], reverse=True):
                f_neg.write(f"{word},{score},neg\n")

    def sopmi(self):
        print('step 1/4:...seg corpus ...')
        seg_data = self.seg_corpus(self.train_path, self.sentiment_path)
        print('step 2/4:...collect cowords ...')
        cowords_list = self.collect_cowords(self.sentiment_path, seg_data)
        print('step 3/4:...compute sopmi ...')
        pmi_dict = self.collect_candiwords(seg_data, cowords_list, self.sentiment_path)
        print('step 4/4:...save candiwords ...')
        self.save_candiwords(pmi_dict, self.candipos_path, self.candineg_path)
        print('finished!')

if __name__ == "__main__":
    sopmier = SoPmi()
    sopmier.sopmi()
