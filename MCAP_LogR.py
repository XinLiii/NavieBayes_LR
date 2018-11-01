import os
import numpy as np
import random
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

##spam = 1, ham = 0
class logistic_reg(object):
    def __init__(self, train_file_ham, train_file_spam, test_file_ham, test_file_spam, improved):
        self.train_file_ham = train_file_ham
        self.train_file_spam = train_file_spam
        self.test_file_ham = test_file_ham
        self.test_file_spam = test_file_spam
        self.improved = improved
        self.iter_num = 100
        self.learning_rate = 0.01
        ##self.record_weight_dic[word,index]
        self.record_weight_dic = {}

    def read_file(self,file_dir):
        one_dic = {}
        with open(file_dir, encoding='utf-8', errors='ignore') as one_file:
            while 1:
                data = one_file.readline()
                if not data:
                    break
                data = data.split(' ')
                for item in data:
                    item = item.strip('\n')
                    if self.improved == 'y':
                        item = ''.join([j for j in item if j.isalpha() and j not in stop_words])
                    else:
                        item = ''.join([j for j in item if j.isalpha()])
                    if len(item) > 0:
                        if item in one_dic:
                            one_dic[item] += 1
                        else:
                            one_dic[item] = 1
            one_file.close()
        return one_dic

    def get_dic(self,file_dir):
        dic = {}
        name_list = os.listdir(file_dir)
        for f in name_list:
            with open(file_dir + f, encoding='utf-8', errors='ignore') as file:
                while 1:
                    data = file.readline()
                    if not data:
                        break
                    data = data.split(' ')
                    for item in data:
                        item = item.strip('\n')
                        if self.improved == 'y':
                            item = ''.join([j for j in item if j.isalpha() and j not in stop_words])
                        else:
                            item = ''.join([j for j in item if j.isalpha()])
                        if len(item) > 0:
                            if item in dic:
                                dic[item] += 1
                            else:
                                dic[item] = 1
            file.close()
        return dic

    def merge_dic(self, dic1, dic2):
        list1, list2 = list(dic1.keys()), list(dic2.keys())
        mergdic = {}
        for i in list1:
            mergdic[i] = dic1[i]
        for i in list2:
            if i in mergdic:
                mergdic[i] += dic2[i]
            else:
                mergdic[i] = dic2[i]
        return mergdic

    def get_raw_data(self):
        spam_documents = self.get_dic(self.train_file_spam)
        ham_documents = self.get_dic(self.train_file_ham)
        ##merge_dic = self.merge_dic(spam_documents, ham_documents)
        self.record_weight_dic = self.merge_dic(spam_documents, ham_documents)
        ##attribute_list = np.array(list(merge_dic.keys()))
        attribute_list = np.array(list(self.record_weight_dic.keys()))
        attribute_list = np.append(attribute_list,'class')
        length_of_data_list = len(attribute_list)
        ##let the value of dictionary record the index of keys in the attribute_list
        for index in range(length_of_data_list-1):
            self.record_weight_dic[attribute_list[index]] = index
        ##record all spam files
        name_list = os.listdir(self.train_file_spam)
        spam_list = np.zeros((len(name_list),length_of_data_list), dtype=float)
        file_num = 0
        for f in name_list:
            tmp_dic = self.read_file(self.train_file_spam + f)
            for key in list(tmp_dic.keys()):
                spam_list[file_num][self.record_weight_dic[key]] = float(tmp_dic[key])
            file_num += 1
        for si in range(len(spam_list)):
            spam_list[si][-1] = 1
        ##record all ham files
        file_num = 0
        name_list = os.listdir(self.train_file_ham)
        ham_list = np.zeros((len(name_list), length_of_data_list), dtype=float)
        for f in name_list:
            tmp_dic = self.read_file(self.train_file_ham + f)
            for key in list(tmp_dic.keys()):
                ham_list[file_num][self.record_weight_dic[key]] = float(tmp_dic[key])
            file_num += 1
        total_data_list = np.vstack((spam_list, ham_list))
        index = list(range(len(total_data_list)))
        random.shuffle(index)
        data_list = total_data_list[0]
        for k in index:
            data_list=np.vstack((data_list,total_data_list[k]))
        return data_list[1:]
        ##return attribute_list


    def LR_training(self, lamb):
        ##data_list.shape(463,9091)
        data_list = self.get_raw_data()
        seed = 0.003
        weight_list, weight_update = np.zeros((len(data_list[0])-1,1)), np.zeros((len(data_list[0])-1,1))
        bias, bias_update = 0, 0
        for index in range(len(weight_list)):
            weight_list[index] = seed
        yl = np.array([y[-1] for y in data_list[0:]])
        yl = np.reshape(yl, [-1, 1])
        xl = np.array([xi[:-1] for xi in data_list[0:]])
        for iter in range(self.iter_num):
            print("iteration: ", iter)
            kkk = np.dot(xl, weight_list) + bias
            for k in range(len(kkk)):
                if kkk[k]>709: kkk[k] = 708
            ##tmp = np.exp(np.dot(xl, weight_list) + bias) / (1 + np.exp(np.dot(xl, weight_list) + bias))
            tmp = np.exp(kkk) / (1 + np.exp(kkk))
            ##weight update
            for i in range(len(weight_list)):
                xl_i = np.array([x[i] for x in data_list])
                xl_i = np.reshape(xl_i, [-1, 1])
                weight_update[i] = weight_list[i]+self.learning_rate*np.sum(xl_i*(yl-tmp))\
                                   -self.learning_rate*lamb*weight_list[i]
            ##bias update

            bias_update = bias+self.learning_rate*np.sum((yl-tmp))-self.learning_rate*lamb*bias
            bias = bias_update
            for items in range(len(weight_list)):
                weight_list[items] = weight_update[items]
        return weight_list, bias

    def test_LR(self, weight, bias):
        test_name_list_spam, test_name_list_ham = os.listdir(self.test_file_spam), os.listdir(self.test_file_ham)
        number_correct_spam = number_correct_ham = 0
        for test_file in test_name_list_spam:
            x_list = np.zeros((len(weight), 1))
            file_name = self.test_file_spam + test_file
            tmp_dic = self.read_file(file_name)
            for key in list(tmp_dic.keys()):
                if key in self.record_weight_dic:
                    x_list[self.record_weight_dic[key]] = tmp_dic[key]
            ##weight.shape = (n,1)  x_list.shape(n,1)
            if sum(np.dot(weight.T, x_list)) + bias > 0:
                number_correct_spam += 1
        for test_file in test_name_list_ham:
            x_list = np.zeros([len(weight), 1])
            file_name = self.test_file_ham + test_file
            tmp_dic = self.read_file(file_name)
            for key in list(tmp_dic.keys()):
                if key in self.record_weight_dic:
                    x_list[self.record_weight_dic[key]] = tmp_dic[key]
            ##weight.shape = (n,1)  x_list.shape(n,1)
            if sum(np.dot(weight.T, x_list)) + bias < 0:
                number_correct_ham += 1
        return (number_correct_spam + number_correct_ham)/(len(test_name_list_spam) + len(test_name_list_ham))