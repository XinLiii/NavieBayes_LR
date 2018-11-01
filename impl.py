from naive_bayes import naiveBayes as nb
from MCAP_LogR import logistic_reg as lr
import sys

improved1 = sys.argv[1]
improved2 = sys.argv[2]
lamb = sys.argv[3]

nb_accuracy = nb(train_file_ham="./train/ham/", train_file_spam="./train/spam/",
               test_file_ham="./test/ham/", test_file_spam="./test/spam/",improved=improved1).test_multinomial_nb()
print("naive bayes accuracy: ", nb_accuracy)
print("")

print("start logistic regression")
lr_impl = lr(train_file_ham="./train/ham/", train_file_spam="./train/spam/",
               test_file_ham="./test/ham/", test_file_spam="./test/spam/",improved=improved2)
lr_weight, lr_bias = lr_impl.LR_training(eval(lamb))
lr_accuracy = lr_impl.test_LR(lr_weight, lr_bias)
print("logistic regression accuracy: ", lr_accuracy)