import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from math import *
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


class RunThread(QThread):
    _signal = pyqtSignal(str)

    def __init__(self):
        super(RunThread, self).__init__()

    def recognition_emo(self, num):
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

    def generate_samples(self, List):
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

    def get_pleasure(self, List):
        if len(List) == 0:
            pleasure = 0
        elif len(List) == 1:
            pleasure = self.recognition_emo(int(List[0] + 2))[0]
        elif len(List) > 1:
            pleasure = self.treat_list(List)
        return pleasure

    def treat_list(self, List):
        sum_list = 0
        for i in range(len(List)):
            sum_list = sum_list + self.recognition_emo(int(List[i] + 2))[0]
        aver = sum_list / len(List)
        return aver

    def GaussianHMM(self, List, X, i):
        i = i + 1
        model = joblib.load("C:/Users/traci/Psycho-Pass/GaussianHMM.pkl")
        hidden_state = model.predict(X)
        x = []
        y = []
        os.chdir("C:/Users/traci/Psycho-Pass/Myfig/")
        pl.figure(i)
        pl.figure(101)
        pl.figure(102)
        pl.figure(103)
        pl.figure(104)
        pl.figure(105)
        for m in range(len(List)):
            x.append(List[m])
            y.append(m)
        if hidden_state[len(hidden_state) - 1] == 0:
            pl.figure(101)
            pl.title("Congratulations! Your're in good state.")
            pl.plot(y, x)
            pl.savefig('f101.png')
            os.chdir("C:/Users/traci/Psycho-Pass/Myfig/State1/")
            f = pl.figure(i)
            f.set_figheight(3.5)
            f.set_figwidth(5)
            pl.title('f' + str(i) + ": Congratulations! Your're in good state.")
            pl.plot(y, x)
        elif hidden_state[len(hidden_state) - 1] == 1:
            pl.figure(102)
            pl.title("Some fluctuations, but it's OK.")
            pl.plot(y, x)
            pl.savefig('f102.png')
            os.chdir("C:/Users/traci/Psycho-Pass/Myfig/State2/")
            f = pl.figure(i)
            f.set_figheight(3.5)
            f.set_figwidth(5)
            pl.title('f' + str(i) + ": Some fluctuations, but it's OK.")
            pl.plot(y, x)
        elif hidden_state[len(hidden_state) - 1] == 2:
            pl.figure(103)
            pl.title("Your state is improving, please stay it.")
            pl.plot(y, x)
            pl.savefig('f103.png')
            os.chdir("C:/Users/traci/Psycho-Pass/Myfig/State3/")
            f = pl.figure(i)
            f.set_figheight(3.5)
            f.set_figwidth(5)
            pl.title('f' + str(i) + ": Your state is improving, please stay it.")
            pl.plot(y, x)
        elif hidden_state[len(hidden_state) - 1] == 3:
            pl.figure(104)
            pl.title("Maybe you need to relax and shift your focus.")
            pl.plot(y, x)
            pl.savefig('f104.png')
            os.chdir("C:/Users/traci/Psycho-Pass/Myfig/State4/")
            f = pl.figure(i)
            f.set_figheight(3.5)
            f.set_figwidth(5)
            pl.title('f' + str(i) + ": Maybe you need to relax and shift your focus.")
            pl.plot(y, x)
        elif hidden_state[len(hidden_state) - 1] == 4:
            pl.figure(105)
            pl.title("Come on! You must adjust your state.")
            pl.plot(y, x)
            pl.savefig('f105.png')
            os.chdir("C:/Users/traci/Psycho-Pass/Myfig/State5/")
            f = pl.figure(i)
            f.set_figheight(3.5)
            f.set_figwidth(5)
            pl.title('f' + str(i) + ": Come on! You must adjust your state.")
            pl.plot(y, x)
        if i < 10:
            pl.savefig('f0' + str(i) + '.png')
        else:
            pl.savefig('f' + str(i) + '.png')

    def del_stopwords(self, seg_sent):
        stopwords = self.read_lines("C:/Users/traci/Psycho-Pass/stop_words.txt")  # 读取停用词表
        new_sent = []   # 去除停用词后的句子
        for word in seg_sent:
            if word in stopwords:
                continue
            else:
                new_sent.append(word)
        return new_sent

    def read_lines(self, filename):
        fp = open(filename, 'r')
        lines = []
        for line in fp.readlines():
            line = line.strip()
            lines.append(line)
        fp.close()
        return lines

    def fenci(self, filename):
        f = open(filename, 'r+', encoding='gbk')
        result＿list = []
        for line in f.readlines():
            seg_list = jieba.cut(line)
            result = []
            for seg in seg_list:
                seg = ''.join(seg.split())
                if(seg != '' and seg != "\n" and seg != "\n\n"):
                    result.append(seg)
            result = self.del_stopwords(result)
            result_list.append(' '.join(result))
        f.close()
        return result＿list

    def svm_predict(self, folder):
        categories = ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11',
                      'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24']

        twenty_train = load_files('C:/Users/traci/Psycho-Pass/seg_file/',
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
        self._signal.emit('Building prefix dict from the default dictionary ...')
        clf = joblib.load("C:/Users/traci/Psycho-Pass/train_model_svm.pkl")
        self._signal.emit('Prefix dict has been built succesfully')
        self._signal.emit('The LibSVM is working ...')
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
                for docs in self.fenci(file):
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

    def run(self):
        self._signal.emit('Loading Offline Data ...')
        self._signal.emit('The Offline Data is loaded')
        self._signal.emit('Start The LibSVM Model ...')
        array = self.svm_predict('C:/Users/traci/Psycho-Pass/data/')
        self._signal.emit('The LibSVM Model is done')
        self._signal.emit('Start The GuassianHMM Model ...')
        self._signal.emit('Loading Model Data ...')
        temp_list = [[0 for col in range(30)] for row in range(50)]
        for i in range(50):
            for j in range(30):
                temp_list[i][j] = self.get_pleasure(array[i][j])
        self._signal.emit('The Model Data is loaded')
        List = []
        self._signal.emit('The GuassianHMM Model is working ...')
        for i in range(50):
            self._signal.emit('Processing NO.' + str(i) + ' data ...')
            X = self.generate_samples(temp_list[i])
            List.append(X)
            self.GaussianHMM(temp_list[i], X, i)
            self._signal.emit('NO.' + str(i + 1) + ' data processing is complete')
        self._signal.emit('The GuassianHMM Model is done')
        self._signal.emit('The System is done')


class Main(QMainWindow):
    path = ''

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)

        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        importAction = QAction('&Import', self)
        importAction.setShortcut('Ctrl+I')
        importAction.setStatusTip('Import data')
        importAction.triggered.connect(self.showDialog)

        runAction = QAction('&Run', self)
        runAction.setShortcut('Ctrl+R')
        runAction.setStatusTip('Run The System')
        runAction.triggered.connect(self.start_treat)

        self.state1Action = QAction('&state1', self)
        self.state1Action.setStatusTip('Visualize State 1')
        # importAction.triggered.connect()

        self.state2Action = QAction('&state2', self)
        self.state2Action.setStatusTip('Visualize State 2')
        # importAction.triggered.connect()

        self.state3Action = QAction('&state3', self)
        self.state3Action.setStatusTip('Visualize State 3')
        # importAction.triggered.connect()

        self.state4Action = QAction('state4', self)
        self.state4Action.setStatusTip('Visualize State 4')
        # importAction.triggered.connect()

        self.state5Action = QAction('&state5', self)
        self.state5Action.setStatusTip('Visualize State 5')
        # state0Action.triggered.connect()

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(importAction)
        fileMenu.addAction(runAction)
        fileMenu.addAction(exitAction)

        visualizationbar = self.menuBar()
        visualizationMenu = visualizationbar.addMenu('&Visualization')
        visualizationMenu.addAction(self.state1Action)
        visualizationMenu.addAction(self.state2Action)
        visualizationMenu.addAction(self.state3Action)
        visualizationMenu.addAction(self.state4Action)
        visualizationMenu.addAction(self.state5Action)

        self.setGeometry(300, 300, 1080, 608)
        self.setWindowTitle('Psycho_Pass')
        self.show()

    def showDialog(self):
        self.path = QFileDialog.getExistingDirectory()

    def start_treat(self):
        self.thread = RunThread()
        self.thread._signal.connect(self.callback)
        self.textEdit.setText('Start The System ...')
        self.thread.start()

    def callback(self, string):
        self.textEdit.append(string)


class State1_Totally(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/f101.png')
        lb = QLabel(self)
        lb.setPixmap(pixmap)
        self.grid.addWidget(lb, 1, 0)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.move(300, 200)
        self.setWindowTitle('State1 Totally')

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State1_Partly(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.folders3 = []
        folder = 'C:/Users/traci/Psycho-Pass/Myfig/State1/'
        os.chdir(folder)
        folders1 = os.listdir()
        folders1.sort()
        for folder in folders1:
            self.folders3 = os.listdir()
            self.folders3.sort()
        if len(self.folders3) >= 6:
            num_page = ceil(len(self.folders3) / 6)
        else:
            num_page = 1
        self.combo1 = QComboBox(self)
        for p in range(num_page):
            self.combo1.addItem("Page " + str(p + 1))
        self.grid.addWidget(self.combo1, 1, 0)
        self.combo1.activated[str].connect(self.onActivated)
        self.move(300, 200)
        self.setWindowTitle('State1 Partly')

    def Page_1(self):
        page = 1
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_2(self):
        page = 2
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_3(self):
        page = 3
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_4(self):
        page = 4
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_5(self):
        page = 5
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_6(self):
        page = 6
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_7(self):
        page = 7
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_8(self):
        page = 8
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State1/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def onActivated(self, text):
        if text == 'Page 1':
            self.Page_1()
        elif text == 'Page 2':
            self.Page_2()
        elif text == 'Page 3':
            self.Page_3()
        elif text == 'Page 4':
            self.Page_4()
        elif text == 'Page 5':
            self.Page_5()
        elif text == 'Page 6':
            self.Page_6()
        elif text == 'Page 7':
            self.Page_7()
        elif text == 'Page 8':
            self.Page_8()

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State2_Totally(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/f102.png')
        lb = QLabel(self)
        lb.setPixmap(pixmap)
        self.grid.addWidget(lb, 1, 0)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.move(300, 200)
        self.setWindowTitle('State2 Totally')

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State2_Partly(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.folders3 = []
        folder = 'C:/Users/traci/Psycho-Pass/Myfig/State2/'
        os.chdir(folder)
        folders1 = os.listdir()
        folders1.sort()
        for folder in folders1:
            self.folders3 = os.listdir()
            self.folders3.sort()
        if len(self.folders3) >= 6:
            num_page = ceil(len(self.folders3) / 6)
        else:
            num_page = 1
        self.combo1 = QComboBox(self)
        for p in range(num_page):
            self.combo1.addItem("Page " + str(p + 1))
        self.grid.addWidget(self.combo1, 1, 0)
        self.combo1.activated[str].connect(self.onActivated)
        self.move(300, 200)
        self.setWindowTitle('State2 Partly')

    def Page_1(self):
        page = 1
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_2(self):
        page = 2
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_3(self):
        page = 3
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_4(self):
        page = 4
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_5(self):
        page = 5
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_6(self):
        page = 6
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_7(self):
        page = 7
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_8(self):
        page = 8
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State2/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def onActivated(self, text):
        if text == 'Page 1':
            self.Page_1()
        elif text == 'Page 2':
            self.Page_2()
        elif text == 'Page 3':
            self.Page_3()
        elif text == 'Page 4':
            self.Page_4()
        elif text == 'Page 5':
            self.Page_5()
        elif text == 'Page 6':
            self.Page_6()
        elif text == 'Page 7':
            self.Page_7()
        elif text == 'Page 8':
            self.Page_8()

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State3_Totally(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/f103.png')
        lb = QLabel(self)
        lb.setPixmap(pixmap)
        self.grid.addWidget(lb, 1, 0)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.move(300, 200)
        self.setWindowTitle('State3 Totally')

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State3_Partly(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.folders3 = []
        folder = 'C:/Users/traci/Psycho-Pass/Myfig/State3/'
        os.chdir(folder)
        folders1 = os.listdir()
        folders1.sort()
        for folder in folders1:
            self.folders3 = os.listdir()
            self.folders3.sort()
        if len(self.folders3) >= 6:
            num_page = ceil(len(self.folders3) / 6)
        else:
            num_page = 1
        self.combo1 = QComboBox(self)
        for p in range(num_page):
            self.combo1.addItem("Page " + str(p + 1))
        self.grid.addWidget(self.combo1, 1, 0)
        self.combo1.activated[str].connect(self.onActivated)
        self.move(300, 200)
        self.setWindowTitle('State3 Partly')

    def Page_1(self):
        page = 1
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_2(self):
        page = 2
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_3(self):
        page = 3
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_4(self):
        page = 4
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_5(self):
        page = 5
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_6(self):
        page = 6
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_7(self):
        page = 7
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_8(self):
        page = 8
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State3/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def onActivated(self, text):
        if text == 'Page 1':
            self.Page_1()
        elif text == 'Page 2':
            self.Page_2()
        elif text == 'Page 3':
            self.Page_3()
        elif text == 'Page 4':
            self.Page_4()
        elif text == 'Page 5':
            self.Page_5()
        elif text == 'Page 6':
            self.Page_6()
        elif text == 'Page 7':
            self.Page_7()
        elif text == 'Page 8':
            self.Page_8()

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State4_Totally(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/f104.png')
        lb = QLabel(self)
        lb.setPixmap(pixmap)
        self.grid.addWidget(lb, 1, 0)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.move(300, 200)
        self.setWindowTitle('State4 Totally')

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State4_Partly(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.folders3 = []
        folder = 'C:/Users/traci/Psycho-Pass/Myfig/State4/'
        os.chdir(folder)
        folders1 = os.listdir()
        folders1.sort()
        for folder in folders1:
            self.folders3 = os.listdir()
            self.folders3.sort()
        if len(self.folders3) >= 6:
            num_page = ceil(len(self.folders3) / 6)
        else:
            num_page = 1
        self.combo1 = QComboBox(self)
        for p in range(num_page):
            self.combo1.addItem("Page " + str(p + 1))
        self.grid.addWidget(self.combo1, 1, 0)
        self.combo1.activated[str].connect(self.onActivated)
        self.move(300, 200)
        self.setWindowTitle('State4 Partly')

    def Page_1(self):
        page = 1
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_2(self):
        page = 2
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_3(self):
        page = 3
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_4(self):
        page = 4
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_5(self):
        page = 5
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_6(self):
        page = 6
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_7(self):
        page = 7
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_8(self):
        page = 8
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State4/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def onActivated(self, text):
        if text == 'Page 1':
            self.Page_1()
        elif text == 'Page 2':
            self.Page_2()
        elif text == 'Page 3':
            self.Page_3()
        elif text == 'Page 4':
            self.Page_4()
        elif text == 'Page 5':
            self.Page_5()
        elif text == 'Page 6':
            self.Page_6()
        elif text == 'Page 7':
            self.Page_7()
        elif text == 'Page 8':
            self.Page_8()

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State5_Totally(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/f105.png')
        lb = QLabel(self)
        lb.setPixmap(pixmap)
        self.grid.addWidget(lb, 1, 0)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.move(300, 200)
        self.setWindowTitle('State5 Totally')

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


class State5_Partly(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.grid = QGridLayout(self)
        self.setLayout(self.grid)
        self.combo = QComboBox(self)
        self.combo.addItem("Totally")
        self.combo.addItem("Partly")
        self.grid.addWidget(self.combo, 0, 0)
        self.folders3 = []
        folder = 'C:/Users/traci/Psycho-Pass/Myfig/State5/'
        os.chdir(folder)
        folders1 = os.listdir()
        folders1.sort()
        for folder in folders1:
            self.folders3 = os.listdir()
            self.folders3.sort()
        if len(self.folders3) >= 6:
            num_page = ceil(len(self.folders3) / 6)
        else:
            num_page = 1
        self.combo1 = QComboBox(self)
        for p in range(num_page):
            self.combo1.addItem("Page " + str(p + 1))
        self.grid.addWidget(self.combo1, 1, 0)
        self.combo1.activated[str].connect(self.onActivated)
        self.move(300, 200)
        self.setWindowTitle('State5 Partly')

    def Page_1(self):
        page = 1
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_2(self):
        page = 2
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_3(self):
        page = 3
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_4(self):
        page = 4
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_5(self):
        page = 5
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_6(self):
        page = 6
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_7(self):
        page = 7
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def Page_8(self):
        page = 8
        for i in range(6):
            if len(self.folders3) - (page - 1) * 6 < i + 1:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/temp.png')
            else:
                pixmap = QPixmap('C:/Users/traci/Psycho-Pass/Myfig/State5/' + self.folders3[i + (page - 1) * 6])
            lb = QLabel(self)
            lb.setPixmap(pixmap)
            if i <= 2:
                self.grid.addWidget(lb, 2, i)
            else:
                self.grid.addWidget(lb, 3, i - 3)

    def onActivated(self, text):
        if text == 'Page 1':
            self.Page_1()
        elif text == 'Page 2':
            self.Page_2()
        elif text == 'Page 3':
            self.Page_3()
        elif text == 'Page 4':
            self.Page_4()
        elif text == 'Page 5':
            self.Page_5()
        elif text == 'Page 6':
            self.Page_6()
        elif text == 'Page 7':
            self.Page_7()
        elif text == 'Page 8':
            self.Page_8()

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    s1 = State1_Totally()
    s2 = State2_Totally()
    s3 = State3_Totally()
    s4 = State4_Totally()
    s5 = State5_Totally()
    s1_p = State1_Partly()
    s2_p = State2_Partly()
    s3_p = State3_Partly()
    s4_p = State4_Partly()
    s5_p = State5_Partly()
    ex.state1Action.triggered.connect(s1.handle_click)
    ex.state2Action.triggered.connect(s2.handle_click)
    ex.state3Action.triggered.connect(s3.handle_click)
    ex.state4Action.triggered.connect(s4.handle_click)
    ex.state5Action.triggered.connect(s5.handle_click)

    def s1onActivated(text):
        if text == 'Totally':
            s1.handle_click()
        else:
            s1_p.handle_click()

    def s2onActivated(text):
        if text == 'Totally':
            s2.handle_click()
        else:
            s2_p.handle_click()

    def s3onActivated(text):
        if text == 'Totally':
            s3.handle_click()
        else:
            s3_p.handle_click()

    def s4onActivated(text):
        if text == 'Totally':
            s4.handle_click()
        else:
            s4_p.handle_click()

    def s5onActivated(text):
        if text == 'Totally':
            s5.handle_click()
        else:
            s5_p.handle_click()
    s1.combo.activated[str].connect(s1onActivated)
    s2.combo.activated[str].connect(s2onActivated)
    s3.combo.activated[str].connect(s3onActivated)
    s4.combo.activated[str].connect(s4onActivated)
    s5.combo.activated[str].connect(s5onActivated)
    sys.exit(app.exec_())
