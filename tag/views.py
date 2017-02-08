from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

# importing sklearn - module
from utility import *

path = '/home/pycon/Desktop/4th-year-project/Ques_Ans_System/Q_A_System/data_sets.csv'
path_classifier ='/home/pycon/Desktop/4th-year-project/Ques_Ans_System/Q_A_System/classifier.pkl'

sentiment_path_classifier ='/home/pycon/Desktop/4th-year-project/Ques_Ans_System/Q_A_System/sentiment.pkl'

sentiment_path ='/home/pycon/Desktop/4th-year-project/Ques_Ans_System/Q_A_System/trainDataFeatures.tsv'
           

d_set = pd.read_csv(path,header=None,names=['tag','title'],dtype=object)

sentiment_df = pd.read_csv(path,header=None,names=['tag','title'],dtype=object)


y_labels = []                                                                  # contains values of d_set['tag']
for i in d_set['tag']:  
    y_labels.append(ast.literal_eval(i))                                       # to remove unicodeed string 
y_labels = [j for i in y_labels for j in i] 
y_labels = list(set(y_labels))



# model training 
class lemmatokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.token_pattern = r"(?u)\b\w\w+\b"       
#         self.wnl = WordNetLemmatizer()
    def __call__(self,doc):                                                     # here, doc is one string sentence
        token_pattern = re.compile(self.token_pattern)
        return [self.stemmer.stem(t) for t in token_pattern.findall(doc)]       # return lambda doc: token_pattern.findall(doc) 
#         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


vect_title = CountVectorizer(max_df=0.5,min_df=5,stop_words='english',tokenizer=lemmatokenizer(),ngram_range=(1,3))

tfidf_vect_title = TfidfVectorizer(smooth_idf=False,max_df=0.5,min_df=5,stop_words='english',tokenizer=lemmatokenizer(),ngram_range=(1,3))


le = preprocessing.LabelEncoder()  
le.fit(y_labels) 
d_set['label_num'] = pd.Series([le.transform(ast.literal_eval(i)) for i in d_set['tag']])
d_set.head()


new_y_labels = d_set['label_num'].values.tolist()

mlb = MultiLabelBinarizer() 
mlb.fit(new_y_labels)

y_tag_dtm = mlb.transform(new_y_labels) 


X_labels = d_set['title'].values.tolist()

vect_title.fit(X_labels)
X_title_dtm = vect_title.transform(X_labels)


tfidf_vect_title.fit(X_labels)
X_title_dtm_tfidf = tfidf_vect_title.transform(X_labels)


## home page
def home(request):
    return render(request,'home.html',{})

## predicting tag here
@csrf_exempt
def predict_tag(request):
    print 'inside predicting tag'
    # test = ["what is ai,lstm and data visualization ?"] 
    rf_clf = joblib.load(path_classifier)
    print rf_clf
    form_data=request.POST.get('form_data','')
    print form_data
    form_data=json.loads(form_data)
    print form_data

    for i in range(len(form_data)):
        current_dict=form_data[i]

        if current_dict['name']=='input-search':
            ques_list=current_dict['value'].split(',')
            # for debugging
            print ques_list
            print '$$$$$$$$$$$$$$'

            test_dtm = tfidf_vect_title.transform(ques_list)                                        # with tfidf

            status = False
            for i in test_dtm.toarray()[0]:
                if (i!=0):
                    status = True
                    break

            ans = rf_clf.predict(test_dtm.toarray())
            ans = mlb.inverse_transform(ans)
            if (len(ans[0])==0 or status==False):
                ans=[["sorry, we can't predict your category!!!"]]
            else:
                ans = le.inverse_transform(ans)
                print type(ans)
            

    search_tag=[]
    for tag in ans[0]:
        search_tag.append(tag)
    # print search_tag      
    return JsonResponse(search_tag,safe=False)  



@csrf_exempt
def predict_sentiment(request):
    print 'inside predicting sentiment'
    # test = ["what is ai,lstm and data visualization ?"] 
    rf_clf = joblib.load(sentiment_path_classifier)
    print rf_clf
    form_data=request.POST.get('form_data','')
    print form_data
    form_data=json.loads(form_data)
    print form_data

    for i in range(len(form_data)):
        current_dict=form_data[i]

        if current_dict['name']=='input-search':
            ques_list=current_dict['value'].split(',')
            # for debugging
            print ques_list
            print '$$$$$$$$$$$$$$'

            
            vocab=['aweosme','interesting','good','nice','terrible','amazing']                                        # with tfidf
            vectorizer=CountVectorizer(vocabulary=vocab)
            test_dtm=vectorizer.fit_transform(ques_list)                                    # with tfidf
                                        # with tfidf

            status = False
            for i in test_dtm.toarray()[0]:
                if (i!=0):
                    status = True
                    break

            ans = rf_clf.predict(test_dtm.toarray())
            ans = mlb.inverse_transform(ans)
            if (len(ans[0])==0 or status==False):
                ans=[["sorry, we can't predict your category!!!"]]
            else:
                ans = le.inverse_transform(ans)
                print type(ans)
            

    search_tag=[]
    for tag in ans[0]:
        search_tag.append(tag)
    # print search_tag      
    return JsonResponse(search_tag,safe=False)      

                                          
    
        







