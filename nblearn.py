import sys
from read_input import load_train_data
from naive_bayes import Naive_Bayes

if __name__=='__main__':
    # argument = sys.argv
    # train_directory =  argument[1]
    train_directory = 'C:\\CSCI544-NaturalLanguageProcessing\\HW1\\train\\'

    # get the classes, reviews for each class and paths of each file from the given directory
    classes, review_list, paths = load_train_data(train_directory)

    naive_bayes = Naive_Bayes(classes)
    processed_reviews = naive_bayes.train_data_preprocessing(review_list)
    naive_bayes.train_model(processed_reviews)

    naive_bayes.create_model()