
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class arg:
    batch_size = -800
    c = 5
    epoch = 400
    lr = 0.1
    inter_print = 1
    gen_policy = "average"
    auto_save = True
    ngram_range=(1,2)
    auto_load = True
    model_path = "./data/model.npy"


class Picture(object):
    def show_train_valid_loss_acc(self, loss, acc, val_loss, val_acc):
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc='lower right')
        plt.figure()

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


class CustomModel(object):
    '''
        set 其中x是原始输入，y是原始label
    '''
    [gen_policy, c, epoch, batch_size, lr] = [
        arg.gen_policy, arg.c, arg.epoch, arg.batch_size, arg.lr]

    def __init__(self, train_set, valid_set, vectorizer):
        self.t_data, self.t_label, self.t_target = vectorizer.transform(
            train_set[0]), train_set[1], self._one_hot(train_set[1])
        self.v_data, self.v_label, self.v_target = vectorizer.transform(
            valid_set[0]), valid_set[1], self._one_hot(valid_set[1])
        self.vectorizer = vectorizer

        self.t_size, self.f_size = self.t_data.shape

        self.cur_epoch = 0
        self.W = np.random.random((self.f_size, self.c))
        self.b = np.random.random(self.c)

        if arg.auto_load:
            self._load_model()

    def _save_model(self):
        np.save(arg.model_path, [self.W, self.b, self.cur_epoch])

    def _load_model(self):
        if os.path.exists(arg.model_path):
            self.W, self.b, self.cur_epoch = np.load(
                arg.model_path, allow_pickle=True)

    def _one_hot(self, y):
        ret = np.zeros((len(y), self.c))
        ret[np.arange(len(y)), y] = 1
        return ret

    def _softmax(self, inp):
        exp = np.exp(np.max(inp)-inp)
        exp /= np.sum(exp, axis=1, keepdims=True)
        return exp

    def _acc(self, src, tat):
        assert len(src) == len(tat)
        return np.sum(src == tat)/len(src)

    def _genresult(self, distribute):
        if self.gen_policy == "average":
            return [np.random.choice(a=self.c, p=dis) for dis in distribute]
        else:
            return np.argmax(distribute, axis=1)

    def _gen_acc(self, dis, label):
        return self._acc(self._genresult(dis), label)

    def forword(self, X, y, label, with_grad=False):
        sample_num = len(y)
        batch_size = sample_num if self.batch_size < 0 else self.batch_size
        predit = []
        loss = []
        for s_idx in range(0, sample_num, batch_size):
            t_idx = min(s_idx+batch_size, sample_num)
            t_batch_size = t_idx - s_idx
            bX = X[s_idx:t_idx]
            bl = label[s_idx:t_idx]
            by = y[s_idx:t_idx]

            b_predit = self._softmax(bX@self.W+self.b)
            b_loss = np.log(b_predit[range(len(b_predit)), bl])

            predit.append(b_predit)
            loss.append(b_loss)

            if with_grad:
                W_grad = (1/t_batch_size)*(bX.T @ (b_predit-by))
                b_grad = (1/t_batch_size)*np.sum(b_predit-by, axis=0)
                self.W = self.W + self.lr*W_grad
                self.b = self.b + self.lr*b_grad

        predit = np.concatenate(predit)
        loss = -np.mean(np.concatenate(loss, axis=0))
        return predit, loss

    def train(self):
        loss_list = []
        valid_loss_list = []
        acc_list = []
        valid_acc_list = []
        for i in range(self.cur_epoch, self.epoch):
            train_predict, train_loss = self.forword(
                self.t_data, self.t_target, self.t_label, with_grad=True)
            valid_predict, valid_loss = self.forword(
                self.v_data, self.v_target, self.v_label, with_grad=False)
            loss_list.append(train_loss)
            acc = self._gen_acc(train_predict, self.t_label)
            acc_list.append(acc)
            valid_loss_list.append(valid_loss)
            valid_acc = self._gen_acc(valid_predict, self.v_label)
            valid_acc_list.append(valid_acc)
            if i % arg.inter_print == 0:
                print("epoch:{}\ttrain_loss:{:.4f}\tvalid_loss:{:.4f}\ttrain_acc:{:.4f}\tvalild_acc:{:.4f}".format(
                    i, train_loss, valid_loss, acc, valid_acc))
                if arg.auto_save:
                    self.cur_epoch = i
                    self._save_model()

        return loss_list, acc_list, valid_loss_list, valid_acc_list

    def predict(self, input):
        data = self.vectorizer(input)
        pred = self._softmax(data@self.W+self.b)
        return self._genresult(pred)


train_file = pd.read_csv("./data/train.csv")
test_file = pd.read_csv("./data/test.csv")

vectorizer = CountVectorizer(ngram_range=arg.ngram_range,stop_words='english')
vectorizer.fit(pd.concat([train_file['Phrase'], test_file['Phrase']]).values)

train_set, valid_set = train_test_split(
    train_file, test_size=0.2, random_state=42, shuffle=True)

train_data = train_set['Phrase'].values
train_label = train_set['Sentiment'].values

valid_data = valid_set['Phrase'].values
valid_label = valid_set['Sentiment'].values


model = CustomModel((train_data, train_label),
                    (valid_data, valid_label), vectorizer)
picture = Picture()
picture.show_train_valid_loss_acc(*model.train())
