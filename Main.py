from sentiment_reader import SentimentCorpus
import numpy as np
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
if __name__ == '__main__':
    start=time.time()
    dataset=SentimentCorpus()
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(dataset.train_X,dataset.train_y.ravel())
    y_pred=logreg.predict(dataset.test_X)

    cnf_matrix = metrics.confusion_matrix(dataset.test_y, y_pred)
    print("Confusion Matrix:")
    print(cnf_matrix)
    Accuracy=metrics.accuracy_score(dataset.test_y, y_pred)
    Precision=metrics.precision_score(dataset.test_y,y_pred)
    Recall=metrics.recall_score(dataset.test_y,y_pred)

    print("Accuracy:",Accuracy)
    print("Precision value:",Precision)
    print("Recall value:",Recall)
    print("F-Score:",((2*Precision*Recall)/(Precision+Recall)))

    end = time.time()
    print("Cpu time:",end-start)
