import pandas as pd
import jieba


class SimpleSentimentAnalysis:
    def __init__(self, pos_words_path='candi_pos.txt', neg_words_path='candi_neg.txt'):
        self.pos_words = self.load_words(pos_words_path)
        self.neg_words = self.load_words(neg_words_path)

    def load_words(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return set([line.strip().split(',')[0] for line in file.readlines()])

    def analyze_text(self, text):
        words = set(jieba.cut(text))
        pos_count = sum(1 for word in words if word in self.pos_words)
        neg_count = sum(1 for word in words if word in self.neg_words)
        return pos_count - neg_count

    def analyze_dataframe(self, df, text_column):
        df['情感分数'] = df[text_column].apply(self.analyze_text)
        df['情感倾向'] = df['情感分数'].apply(lambda x: '积极' if x > 0 else ('消极' if x < 0 else '中性'))
        return df


def read_excel(file_path):
    return pd.read_excel(file_path)


def main():
    # 文件路径
    file_path = '用户.xlsx'

    # 初始化情感分析器
    analyzer = SimpleSentimentAnalysis(pos_words_path='candi_pos.txt', neg_words_path='candi_neg.txt')

    # 读取数据
    df = read_excel(file_path)

    # 假设评论列名为'发言文本'
    comment_column = '发言文本'

    # 进行情感分析
    df = analyzer.analyze_dataframe(df, comment_column)

    # 可以选择将结果保存回新的Excel文件
    df.to_excel('分析后的用户数据.xlsx', index=False)


if __name__ == "__main__":
    main()
