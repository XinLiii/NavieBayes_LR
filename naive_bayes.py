from math import log
import os
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


class naiveBayes(object):
    def __init__(self, train_file_ham, train_file_spam, test_file_ham, test_file_spam, improved):
        self.train_file_ham = train_file_ham
        self.train_file_spam = train_file_spam
        self.test_file_ham = test_file_ham
        self.test_file_spam = test_file_spam
        self.improved = improved

    def improved_get_dic(self,file_dir):
        dic = {}
        name_list = os.listdir(file_dir)
        for f in name_list:
            dir = file_dir + f
            with open(dir, encoding='utf-8', errors='ignore') as file:
                while 1:
                    data = file.readline()
                    if not data:
                        break
                    data = data.split(' ')
                    for item in data:
                        item = item.strip('\n')
                        item = ''.join([j for j in item if j.isalpha() and j not in stop_words])
                        if len(item) > 0:
                            if item in dic:
                                dic[item] += 1
                            else:
                                dic[item] = 1
            file.close()
        return dic

    def get_dic(self,file_dir):
        dic = {}
        name_list = os.listdir(file_dir)
        for f in name_list:
            dir = file_dir + f
            with open(dir, encoding='utf-8', errors='ignore') as file:
                while 1:
                    data = file.readline()
                    if not data:
                        break
                    data = data.split(' ')
                    for item in data:
                        item = item.strip('\n')
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

    def train_multinomial_nb(self):
        condprob_spam_dic, condprob_ham_dic = {}, {}
        ##two word vocabularies of spam and ham
        if self.improved == 'y':
            dic_spam, dic_ham = self.improved_get_dic(self.train_file_spam), self.improved_get_dic(self.train_file_ham)
        else:
            dic_spam, dic_ham = self.get_dic(self.train_file_spam), self.get_dic(self.train_file_ham)
        ##merge_dic contains all keys of dic_spam and dic_ham
        merge_dic = self.merge_dic(dic_spam, dic_ham)
        length_of_merge_dic = len(list(merge_dic))
        ##the number of spam docs and the number of ham docs
        num_train_spam, num_train_ham = len(os.listdir(self.train_file_spam)), len(os.listdir(self.train_file_ham))
        ##the total number of training docs
        num_train = num_train_spam + num_train_ham
        ##two priors
        prior_spam, prior_ham = num_train_spam/num_train, num_train_ham/num_train
        ##the total number of words in all spam docs and ham docs
        total_num_word_spam, total_num_word_ham = sum(dic_spam.values()), sum(dic_ham.values())
        for key in list(merge_dic.keys()):
            num_key_in_spam = dic_spam.get(key,0)
            num_key_in_ham = dic_ham.get(key,0)
            condprob_spam_dic[key] = (num_key_in_spam+1)/(total_num_word_spam+length_of_merge_dic)
            condprob_ham_dic[key] = (num_key_in_ham+1)/(total_num_word_ham+length_of_merge_dic)

        return prior_spam, prior_ham, condprob_spam_dic, condprob_ham_dic, merge_dic

    def apply_multinomial_nb(self, spam_prior, ham_prior, condprob_spam, condprob_ham, d):
        one_dic = {}
        with open(d, encoding='utf-8', errors='ignore') as one_file:
            while 1:
                data = one_file.readline()
                if not data:
                    break
                data = data.split(' ')
                for item in data:
                    item = item.strip('\n')
                    item = ''.join([j for j in item if j.isalpha()])
                    if len(item) > 0:
                        if item in one_dic:
                            one_dic[item] += 1
                        else:
                            one_dic[item] = 1
            one_file.close()
        score_spam = log(spam_prior)
        score_ham = log(ham_prior)
        for word in list(one_dic.keys()):
            num_of_word = one_dic.get(word,0)
            score_spam += num_of_word * log(condprob_spam.get(word,1))
            score_ham += num_of_word * log(condprob_ham.get(word,1))
        return score_spam-score_ham


    def test_multinomial_nb(self):
        prior_spam, prior_ham, condprob_spam_dic, condprob_ham_dic, merge_dic = self.train_multinomial_nb()
        test_name_list_spam, test_name_list_ham = os.listdir(self.test_file_spam), os.listdir(self.test_file_ham)
        number_correct_spam = number_correct_ham = 0

        for test_file in test_name_list_spam:
            file_name = self.test_file_spam + test_file
            argmax_value1 = self.apply_multinomial_nb(prior_spam, prior_ham,
                                                     condprob_spam_dic, condprob_ham_dic, file_name)
            if argmax_value1 > 0:
                number_correct_spam += 1

        for test_file in test_name_list_ham:
            file_name = self.test_file_ham + test_file
            argmax_value2 = self.apply_multinomial_nb(prior_spam, prior_ham,
                                                     condprob_spam_dic, condprob_ham_dic, file_name)
            if argmax_value2 < 0:
                number_correct_ham += 1
        """
        print(number_correct_spam,number_correct_ham)
        print(len(test_name_list_spam),len(test_name_list_ham))
        """
        accuracy = (number_correct_spam + number_correct_ham)/(len(test_name_list_spam) + len(test_name_list_ham))
        return accuracy



