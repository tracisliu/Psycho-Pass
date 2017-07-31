import numpy as np
import emotions as em
import pylab as pl
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn import svm
import jieba
from sklearn.datasets import load_files


def recognition_emo(num):
    if num == 1:
        return em.Silent
    elif num == 2:
        return em.Admiration
    elif num == 3:
        return em.Anger
    elif num == 4:
        return em.Disliking
    elif num == 5:
        return em.Disappointment
    elif num == 6:
        return em.Distress
    elif num == 7:
        return em.Fear
    elif num == 8:
        return em.FearsConfirmed
    elif num == 9:
        return em.Gloating
    elif num == 10:
        return em.Gratification
    elif num == 11:
        return em.Gratitude
    elif num == 12:
        return em.HappyFor
    elif num == 13:
        return em.Hate
    elif num == 14:
        return em.Hope
    elif num == 15:
        return em.Joy
    elif num == 16:
        return em.Liking
    elif num == 17:
        return em.Love
    elif num == 18:
        return em.Pity
    elif num == 19:
        return em.Pride
    elif num == 20:
        return em.Relief
    elif num == 21:
        return em.Remorse
    elif num == 22:
        return em.Reproach
    elif num == 23:
        return em.Resentment
    elif num == 24:
        return em.Satisfaction
    elif num == 25:
        return em.Shame


def generate_samples(List):
    X = np.zeros(((len(List) - 2), 3))
    for i in range(len(List) - 2):
        temp = np.array([List[i + 2], List[i + 1], List[i]])
        feature_1 = List[i + 2] - List[i]
        feature_2 = np.mean(temp)
        feature_3 = np.std(temp)
        X[i, 0] = feature_1
        X[i, 1] = feature_2
        X[i, 2] = feature_3
    return X


def get_pleasure(List):
    if len(List) == 0:
        pleasure = 0
    elif len(List) == 1:
        pleasure = recognition_emo(int(List[0] + 2))[0]
    elif len(List) > 1:
        pleasure = treat_list(List)
    return pleasure


def treat_list(List):
    sum_list = 0
    for i in range(len(List)):
        sum_list = sum_list + recognition_emo(int(List[i] + 2))[0]
    aver = sum_list / len(List)
    return aver


def GaussianHMM(List, X, i):
    i = i + 1
    model = joblib.load("C:/Users/traci/Psycho-Pass/GaussianHMM.pkl")
    hidden_state = model.predict(X)
    x = []
    y = []
    os.chdir("C:/Users/traci/Psycho-Pass/Myfig/")
    pl.figure(i)
    for m in range(len(List)):
        x.append(List[m])
        y.append(m)
    if hidden_state[len(hidden_state) - 1] == 0:
        pl.figure(i)
        pl.title("Congratulations! Your're in good state.")
        pl.plot(y, x)
    elif hidden_state[len(hidden_state) - 1] == 1:
        pl.figure(i)
        pl.title("Some fluctuations, but it's OK.")
        pl.plot(y, x)
    elif hidden_state[len(hidden_state) - 1] == 2:
        pl.figure(i)
        pl.title("Your state is improving, please stay it.")
        pl.plot(y, x)
    elif hidden_state[len(hidden_state) - 1] == 3:
        pl.figure(i)
        pl.title("Maybe you need to relax and shift your focus.")
        pl.plot(y, x)
    elif hidden_state[len(hidden_state) - 1] == 4:
        pl.figure(i)
        pl.title("Come on! You must adjust your state.")
        pl.plot(y, x)
    if i < 10:
        pl.savefig('f0' + str(i) + '.png')
    else:
        pl.savefig('f' + str(i) + '.png')


def del_stopwords(seg_sent):
    stopwords = read_lines(
        "C:/Users/traci/Psycho-Pass/stop_words.txt")  # 读取停用词表
    new_sent = []   # 去除停用词后的句子
    for word in seg_sent:
        if word in stopwords:
            continue
        else:
            new_sent.append(word)
    return new_sent


def read_lines(filename):
    fp = open(filename, 'r')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines


def fenci(filename):
    f = open(filename, 'r+', encoding='gbk')
    result＿list = []
    for line in f.readlines():
        seg_list = jieba.cut(line)
        result = []
        for seg in seg_list:
            seg = ''.join(seg.split())
            if(seg != '' and seg != "\n" and seg != "\n\n"):
                result.append(seg)
        result = del_stopwords(result)
        result_list.append(' '.join(result))
    f.close()
    return result＿list


def svm_predict(folder):
    categories = ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11',
                  'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24']

    twenty_train = load_files('seg_file/',
                              categories=categories,
                              load_content=True,
                              encoding='gbk',
                              decode_error='strict',
                              shuffle=True, random_state=42)
    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(twenty_train.data))
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    clf = joblib.load("C:/Users/traci/Psycho-Pass/train_model_svm.pkl")
    list_data = []
    os.chdir(folder)
    folders1 = os .listdir()
    folders1.sort()
    for folder in folders1:
                # print(folder)
        os.chdir(folder)
        list_files = []
        folders2 = os.listdir()
        folders2.sort()
        # print(folders2)
        for file in folders2:
            list_file = []
            for docs in fenci(file):
                                # print(docs)
                if (not docs):
                    pass
                else:
                    docs_new = []
                    docs_new.append(docs)
                    X_new_counts = vectorizer.transform(list(docs_new))
                    X_new_tfidf = transformer.transform(X_new_counts)
                    # predict the target of testing samples
                    result = clf.predict(X_new_tfidf)
                    list_file.append(result)
            list_files.append(list_file)
        os.chdir('..')
        list_data.append(list_files)
    return list_data


def Psycho_Pass():
    array = svm_predict('data/')
    temp_list = [[0 for col in range(30)] for row in range(50)]
    for i in range(50):
        for j in range(30):
            temp_list[i][j] = get_pleasure(array[i][j])
    List = []
    for i in range(50):
        X = generate_samples(temp_list[i])
        List.append(X)
        GaussianHMM(temp_list[i], X, i)

if __name__ == '__main__':
    array = svm_predict('data/')
    temp_list = [[0 for col in range(30)] for row in range(50)]
    for i in range(50):
        for j in range(30):
            temp_list[i][j] = get_pleasure(array[i][j])
    List = []
    for i in range(50):
        X = generate_samples(temp_list[i])
        List.append(X)
        GaussianHMM(temp_list[i], X, i)
