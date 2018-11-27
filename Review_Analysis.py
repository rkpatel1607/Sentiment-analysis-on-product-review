
from util import *

## ------------------------------------------------ Task-1 -------

def load(File):

    line_count = 0
    KEYS = []
    VALUES = []

    # Creating output file to write each user review in Dictionary format
    F2 = open("Amazon_Dict_Reviews.txt", "w")

    # Reading original Amazon Review data file
    F1 = open(File, "r")
    for line in F1:
        try:
            key, val = line.rstrip().split(": ")

            KEYS.append(key)
            if len(val) < 18:
                VALUES.append(val)
            else:
                VALUES.append(stopwords_removal(val))
            line_count += 1

            Dict = {}
            # Each review is set of 8 liens
            if line_count % 8 == 0:
                for num in range(len(KEYS)):
                    Dict[KEYS[num]] = VALUES[num]
                F2.writelines(json.dumps(Dict) + '\n')
        except:
            continue

def stopwords_removal(current_line):

    # Setting the value for Rating key @ Global Dictionary
    if current_line.__contains__(" out of 5 stars"):
        current_line = current_line.split()[0]
        return current_line

    # Setting the value for review key for Global Dictionary
    else:
        updated_line = ' '
        for Word in current_line.lower().split():
            if Word not in stopwords:
                updated_line += Word + ' '
        return updated_line.strip()

## ------------------------------------------------ Task-2 -------

def Dict_Formation(Temp_Dict):

    # Formation of Global Dict for single words
    # Each word have unique dictionary which has its Rating score count per Rating

    for WORD in Temp_Dict:
        if WORD not in stopwords:
            # if word is already in Dictionary, increment its rating count
            if WORD in Global_dict.keys():
                Global_dict[WORD][Rating] += 1
            # if word is not in Dict, initialize all rating for it with 0
            #   Add count of 1 to word according to its Rating group
            else:
                Global_dict[WORD] = {'1.0': 0, '2.0': 0, '3.0': 0, '4.0': 0, '5.0': 0}
                Global_dict[WORD][Rating] = 1

def NOA_NOV_generation(RVW):

    # Defining local variables for Dictionary updates
    NOA_local = [];    NOV_local = []

    for i in range(len(RVW) - 3):
        if RVW[i][1] in ADV:
            if RVW[i+1][0] in Negative_Pre:
                if RVW[i + 1][1] in ADJ:
                    NOA.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
                    NOA_local.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
                if RVW[i + 1][1] in VERB:
                    NOV.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
                    NOV_local.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
                elif RVW[i + 2][1] in ADJ:
                    NOA.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))
                    NOA_local.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))
                if RVW[i + 2][1] in VERB:
                    NOV.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))
                    NOV_local.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))

    return NOA_local, NOV_local

def Cal_Sentiment_Score(Dict):

    # Calculating Gamma Values and Sentiment score for each word

    SS = {}     # Temporary Dict of Scores
    for Key in Dict:
        SumNum = 0
        SumDiv = 0

        # Individual Dictionary for each word
        RatingDict = Global_dict[Key]
        # For each rating group...
        for rating in RatingDict:

            # Checking and finding gamma value, to stop equation returning infinite value
            if Global_dict[Key][rating] == 0:
                gamma = 1
                SumDiv += gamma
            else:
                gamma = (Global_dict[Key]['5.0']) / (Global_dict[Key][rating])
                SumDiv += (gamma * Global_dict[Key][rating])

            # Calculation of numerator and denominator
            f = float(rating)
            SumNum += (f * gamma * Global_dict[Key][rating])

        # updating scores for each word with its sentiment score
        SS[Key] = SumNum / SumDiv

    # returning final sentiment scores
    return SS

def Global_30Plus():

    # Creation of dictionary which has words which occurred more then 30 times
    Temp_Global_Dict = []
    # occurrenceFile = open("occurrenceFile.txt", "w")
    for occurrence in Global_dict:
        rating = Global_dict[occurrence]
        if (rating['1.0'] + rating['2.0'] + rating['3.0'] + rating['4.0'] + rating['5.0']) < 30:
            Temp_Global_Dict.append(occurrence)
            # occurrenceFile.writelines(json.dumps({occurrence: Global_dict[occurrence]}) + '\n')

    # removing less occurred words
    for word in Temp_Global_Dict:
        Global_dict.pop(word)

def COUNT(R, Count1, Count2, Count3, Count4, Count5):

    # Conversion of string value of rating to float
    rate = float(R)

    # incrementing count of rating variable based on occurrence
    if rate == 1.0:
        Count1 += 1
    elif rate == 2.0:
        Count2 += 1
    elif rate == 3.0:
        Count3 += 1
    elif rate == 4.0:
        Count4 += 1
    else:
        Count5 += 1

    return Count1, Count2, Count3, Count4, Count5

def PlotBar(C1, C2, C3, C4, C5):

    # All Rating value's final count for before plotting
    Count = [C1, C2, C3, C4, C5]
    stars = ('1-star', '2-star', '3-star', '4-star', '5-star')
    y_pos = np.arange(len(stars))

    plt.title('Review Categories')
    plt.bar(y_pos, Count, align='center')
    plt.xticks(y_pos, stars)
    plt.ylabel('Reviews')
    plt.show()

## ------------------------------------------------ Task-3 -------

def initialize_roc():
    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves for SVM, NB, RF")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

def F1_plots(F1_Sc):

    names = ['SVM', 'NB', 'RF']
    plt_f1.bar(np.arange(len(names)), F1_Sc, align="center")
    plt_f1.xlabel("Classifier")
    plt_f1.ylabel("F1-score")
    plt_f1.title("F1 Scores & Classifiers")
    plt_f1.xticks(np.arange(len(names)), names)
    plt_f1.show()

## ------------------------------------------------------ Main -------

if __name__ == '__main__':

    start_time = time.time()
    print "Start time: ", str(datetime.now())

    ## ---------------------------------------------------- Task-1 -------
    FileName = "amazon_total.txt"
    # load(FileName)

    ## ---------------------------------------------------- Task-2 -------
    New_File = "Amazon_Dict_Reviews.txt"
    F = open(New_File, "r")
    X = []
    y = []

    count = 0
    for Line in F:
        count += 1
        data = json.loads(Line)

        # Fetching required data from Reviews
        Rating = data['rating'].encode('utf-8')
        Count_1, Count_2, Count_3, Count_4, Count_5 = COUNT(Rating, Count_1, Count_2, Count_3, Count_4, Count_5)

        Review = data['review'].encode('utf-8')
        Review_token = nltk.tokenize.word_tokenize(Review)

        # Generation of X for Models
        X.append(Review)

        # Generation of Y for Models
        Intensity = SentimentIntensityAnalyzer()
        answers = Intensity.polarity_scores(Review)
        Positivity, Negativity = answers['pos'], answers['neg']

        if Positivity > Negativity:
            y.append(1)
        else:
            y.append(0)

        # Dictionary generation from tokenize words
        Dict_Formation(Review_token)
        # generating tuples using POS Tagger; format: ( word , Token )
        RVWs = nltk.pos_tag(Review_token)

        # Generation of NOA and NOV list POS tagged words
        NOA_Local, NOV_Local = NOA_NOV_generation(RVWs)
        Dict_Formation(NOA_Local)
        Dict_Formation(NOV_Local)

        # if count == 50000:
        #     break

    PlotBar(Count_1, Count_2, Count_3, Count_4, Count_5)
    # print Count_1, Count_2, Count_3, Count_4, Count_5

    Global_30Plus()     # Checking each word/phrase occurrence

    # Calculation of Gamma value and Sentiment scores
    Sentiment_Score = Cal_Sentiment_Score(Global_dict)
    # print "Sentiment_Score", Sentiment_Score

    All_points = []
    for k in Sentiment_Score:
        All_points.append(Sentiment_Score[k])

    plt.boxplot(All_points)
    plt.title("sentiment scores")
    plt.show()

    plt.hist(All_points)
    plt.show()

    ## ---------------------------------------------------- Task-3 ------------------------------------

    # Count Vectorizer generation from X
    Vectorizer = CountVectorizer(min_df=1, tokenizer=lambda doc: doc, lowercase=False)
    Final_X = Vectorizer.fit_transform(X).toarray()

    # generation of training and testing data set from input data
    X_train, X_test, y_train, y_test = train_test_split(Final_X, np.array(y), test_size=0.35)

    # 10-fold cross validation with each classifier call
    def F1_calculation(clf):
        scores = cross_val_score(clf, Final_X, y, cv=10, scoring='f1_micro')
        print "f1-micro =>", scores.mean()
        return scores.mean()

    # Classifier
    svm_classifier = svm.SVC()
    nb_classifier = naive_bayes.MultinomialNB()
    rf_classifier = RandomForestClassifier()

    # F1-macro score calculation
    F1_Scores = [F1_calculation(svm_classifier),
                 F1_calculation(nb_classifier),
                 F1_calculation(rf_classifier)]

    F1_plots(F1_Scores)

    initialize_roc()
    # --------------------------------------------------------------------------------------------------

    svm_fit = svm_classifier.fit(X_train, y_train).decision_function(X_test)
    predicted_label_svm = svm_classifier.predict(X_test)
    Acc_svm = metrics.accuracy_score(y_test, predicted_label_svm)

    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_fit)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
    # --------------------------------------------------------------------------------------------------

    nb_fit = nb_classifier.fit(X_train, y_train)
    predicted_label_nb = nb_classifier.predict(X_test)
    Acc_nb = metrics.accuracy_score(y_test, predicted_label_nb)

    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_fit.predict(X_test))
    roc_auc_nb = auc(fpr_nb, tpr_nb)
    plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label='NB ROC curve (area = %0.2f)' % roc_auc_nb)
    # --------------------------------------------------------------------------------------------------

    rf_fit = rf_classifier.fit(X_train, y_train)
    predicted_label_rf = rf_classifier.predict(X_test)
    Acc_rf = metrics.accuracy_score(y_test, predicted_label_rf)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_fit.predict(X_test))
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='RF ROC curve (area = %0.2f)' % roc_auc_rf)
    # --------------------------------------------------------------------------------------------------

    plt.legend(loc="lower right")
    plt.show()

    # print " Accuracy @ SVM = ", Acc_svm
    # print " Accuracy @ NB = ", Acc_nb
    # print " Accuracy @ RF = ", Acc_rf
    # print " Area under ROC curve for SVM is: ", roc_auc_svm
    # print " Area under ROC curve for NB is: ", roc_auc_nb
    # print " Area under ROC curve for RF is: ", roc_auc_rf
    # --------------------------------------------------------------------------- END ----------------

    print "End time: ", str(datetime.now())

    end_time = time.time()
    time = float("{0:.2f}".format((end_time - start_time)/60))
    print "Running time =", time, " Min"
