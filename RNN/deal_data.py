import codecs
import collections
from operator import itemgetter
import sys

MODE = "PTB_TEST"    # 将MODE设置为"PTB_TRAIN", "PTB_VALID", "PTB_TEST", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB_TRAIN":        # PTB训练数据
    RAW_DATA = "./data/ptb.train_done.txt"  # 训练集数据文件
    VOCAB = "./data/done/ptb.vocab"                                 # 词汇表文件
    OUTPUT_DATA = "./data/done/ptb.train_done"                           # 将单词替换为单词编号后的输出文件
elif MODE == "PTB_VALID":      # PTB验证数据
    RAW_DATA = "./data/ptb.valid.txt"
    VOCAB = "./data/done/ptb.vocab"
    OUTPUT_DATA = "./data/done/ptb.valid"
elif MODE == "PTB_TEST":       # PTB测试数据
    RAW_DATA = "./data/ptb.test.txt"
    VOCAB = "./data/done/ptb.vocab"
    OUTPUT_DATA = "./data/done/ptb.test"
elif MODE == "TRANSLATE_ZH":   # 中文翻译数据
    RAW_DATA = "../datasets/TED_data/train_done.txt.zh"
    VOCAB = "zh.vocab"
    OUTPUT_DATA = "train_done.zh"
elif MODE == "TRANSLATE_EN":   # 英文翻译数据
    RAW_DATA = "../datasets/TED_data/train_done.txt.en"
    VOCAB = "en.vocab"
    OUTPUT_DATA = "train_done.en"

def serial_number():
    # 统计单词出现的频率
    counter = collections.Counter()
    with codecs.open(RAW_DATA, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 稍后我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表
    sorted_words = ["<eos>"] + sorted_words

    with codecs.open(VOCAB, 'w', "utf-8") as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")


def word2num():
    # 读取词汇表，并建立词汇到单词编号的映射。
    with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了不在词汇表内的低频词，则替换为"unk"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(RAW_DATA, "r", "utf-8")
    fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()


if __name__ == '__main__':
    serial_number()
    word2num()