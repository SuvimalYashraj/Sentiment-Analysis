import re
import numpy as np
from collections import Counter
from pickle import dump

# list of stop words
stop_words = [
    "a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at"
     ,"be","because","been","before","being","below","between","both","but","by","can","could","couldn","couldn't"
     ,"d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further"
     ,"had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his"
     ,"how","he'd","he'll","he's","here's","how's"
     ,"i","if","in","into","is","isn","isn't","it","it's","its","itself","i'd","i'll","i'm","i've","just","ll"
     ,"m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now"
     ,"o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re"
     ,"s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such"
     ,"t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too"
     ,"under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why"
     ,"will","with","won","won't","wouldn","wouldn't"
     ,"y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
     ,"let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's"
     ,"when's","where's","who's","why's","would","able"
     ,"accordance","according","accordingly","across","act","actually","adj","afterwards"
     ,"ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow"
     ,"anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside"
     ,"ask","asking","auth","available","away","b","back","became","become","becomes","becoming","beforehand","begin"
     ,"beginning","beginnings","begins","behind","beside","besides","beyond","brief","briefly","c","ca","came"
     ,"cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt"
     ,"date","different","done","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end"
     ,"ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except"
     ,"f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore"
     ,"g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed"
     ,"hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie"
     ,"im","inc","indeed","index","information","instead","invention","inward","itd"
     ,"it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","last","lately","later","latter","latterly"
     ,"least","less","lest","let","lets","line","little","'ll","look","looking","looks","ltd","made","mainly"
     ,"make","makes","many","may","maybe","means","meantime","meanwhile","mg","might","million","miss","ml","moreover"
     ,"mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","near","nearly","need","needs"
     ,"nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted"
     ,"nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","others"
     ,"otherwise","outside","overall","owing","p","page","pages","part","past","per","perhaps","placed"
     ,"please","possible","possibly","potentially","predominantly","present","previously","primarily","probably"
     ,"promptly","proud","provides","put","q","que","quite","ran","rather","really","recent"
     ,"recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted"
     ,"resulting","results","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed"
     ,"seeming","seems","seen","self","sent","seven","several","shall","shed","shes","show","showed","shown","showns"
     ,"shows","similar","similarly","since","six","slightly","somebody","somehow","someone"
     ,"somethan","something","sometime","sometimes","somewhat","somewhere","soon","specifically","specified","specify"
     ,"specifying","still","stop","sub","suggest","sup","sure","take"
     ,"taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered"
     ,"therefore","therein","there'll","thereof","thereupon","there've","theyd","theyre","think"
     ,"thou","though","thoughh","thousand","throug","thru","thus","til","tip","together","took","toward","towards"
     ,"tried","tries","try","trying","ts","twice","two","u","un","unless","unlike","unlikely","unto"
     ,"upon","ups","us","use","used","usefulness","uses","using","usually","v","various","'ve"
     ,"via","viz","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll"
     ,"whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim"
     ,"whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","wish","within","without","wont"
     ,"words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear"
     ,"associated","c'mon","c's","cant","clearly","consequently"
     ,"consider","considering","corresponding","course","currently","described","example"
     ,"going","greetings","hello","hopefully","inasmuch","indicate","indicated","indicates","inner","insofar","it'd"
     ,"keep","keeps","novel","presumably","second","secondly","sure","t's","third","three","wonder"    
]

class Naive_Bayes():
    def __init__(self, classes) -> None:
        self.classes = classes

        # initialize logprior[class]: stores the prior probability for all the class
        self.logprior = [None] * len(classes)

        # initialize loglikelihood[class][word] stores the likelihood probability for a word given all the classes
        self.loglikelihood = [None] * len(classes)
    
    def train_data_preprocessing(self, reviews_class_wise):
        ''' convert to lower case and remove punctuations '''
        review_class_no_punctuations = []
        for review_class in reviews_class_wise:
            review_no_punctuations = []
            for review in review_class:
                review_no_punctuations.append(review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").
                        replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[","").replace("]","").replace("\\","").
                        replace("*", "").replace("(", "").replace(")", "").replace("/", "").replace("\"",""))
            review_class_no_punctuations.append(review_no_punctuations)

        ''' remove stop words '''
        review_class_no_stop_words = []
        for review_class in review_class_no_punctuations:
            review_no_stop_words = []
            for review in review_class:
                review_words = review.split()
                reomove_stop_words = [word for word in review_words if word.lower() not in stop_words]
                review_no_stop_words.append(' '.join(reomove_stop_words))
            review_class_no_stop_words.append(review_no_stop_words)

        ''' apply stemming '''
        review_class_stemming = []
        for review_class in review_class_no_stop_words:
            review_stemming = []
            for review in review_class:
                review_split = review.split()
                stemmed = [re.sub('(ing|ed|al|ly)$', '', w) for w in review_split]
                review_stemming.append(' '.join(stemmed))
            review_class_stemming.append(review_stemming)

        return review_class_stemming

    def test_data_preprocessing(self, review_class):
        ''' convert to lower case and remove punctuations '''
        review_no_punctuations = []
        for review in review_class:
            review_no_punctuations.append(review.lower().replace(",", "").replace(".", "").replace("!", "").replace("-","").
                        replace("?", "").replace(";", "").replace(":", "").replace("\\n", "").replace("[","").replace("]","").replace("\\","").
                        replace("*", "").replace("(", "").replace(")", "").replace("/", "").replace("\"",""))

        ''' remove stop words '''
        review_no_stop_words = []
        for review in review_no_punctuations:
            review_words = review.split()
            reomove_stop_words = [word for word in review_words if word.lower() not in stop_words]
            review_no_stop_words.append(' '.join(reomove_stop_words))

        ''' apply stemming '''
        review_stemming = []
        for review in review_no_stop_words:
            review_split = review.split()
            stemmed = [re.sub('(ing|ed|al|ly)$', '', w) for w in review_split]
            review_stemming.append(' '.join(stemmed))

        return review_stemming

    def train_model(self, reviews):
        # find number of documents present for each class
        review_per_class = []
        for review in reviews:
            review_per_class.append(len(review))
        
        # total reviews available
        total_reviews = sum(review_per_class)

        # create a vocabulary set
        self.vocabulary = set()
        for review_class in reviews:
            for review in review_class:
                words = review.split()
                for word in words:
                    if word in self.vocabulary:
                        continue
                    else:
                        self.vocabulary.add(word)
        vocabulary_size = len(self.vocabulary)

        word_counter_per_class = []

        for ci in range(len(self.classes)):
            # compute P(class)
            self.logprior[ci] = np.log(review_per_class[ci]/ total_reviews)

            # find the frequency of each word in class ci and total number of words in class ci
            counter = Counter()
            total_words_in_class = 0
            for review in reviews[ci]:
                words = review.split()
                total_words_in_class += len(words)
                counter += Counter(words)
            
            word_counter_per_class.append(counter)
            denominator = total_words_in_class + vocabulary_size

            # dictionary to store the probabilities for each word in ci where key is the word and value is the probability
            dic = {}
            #Compute P(word|ci)
            for word in self.vocabulary:
                numerator = word_counter_per_class[ci][word] + 1
                dic[word] = np.log((numerator) / (denominator))
            self.loglikelihood[ci] = dic

    ''' write the logprior and loglikelihood probabilities and the vocabulary in a file '''
    def create_model(self):
        with open('nbmodel.txt', 'w') as f:
            f.write("%s\n" % len(self.classes))
            for item in self.logprior:
                f.write("%s\n" % item)
            c = 0
            for item in self.loglikelihood:
                class_name = self.classes[c]
                class1, class2 = class_name.split('.')
                class1 = class1.split('_')[0]
                class2 = class2.split('_')[0]
                f.write("%s.%s\n" % (class1,class2))
                c+=1
                for key, value in item.items(): 
                    f.write('%s:%s\n' % (key, value))
        
        with open('vocabulary.pkl','wb') as f:
            dump(self.vocabulary,f)