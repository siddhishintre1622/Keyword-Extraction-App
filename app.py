from audioop import error
from idlelib.iomenu import errors

from flask import Flask,request,render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from pygments.lexers.srcinfo import keywords

nltk.download('stopwords')
import nltk
nltk.download('punkt_tab')
import pickle

app = Flask(__name__)
cv = pickle.load(open('count_vectorizer.pkl','rb'))
feature_names = pickle.load(open('feature_names.pkl','rb'))
tfidf_transformer = pickle.load(open('tfidf.pkl','rb'))




stopwords = stopwords.words('english')
new_words = ['fig','figure','image','sample','using','show','result','large','also','one','two','three','five','six','seven','eight','nine','ten']
stopwords = stopwords + new_words


def preprocessing_text(txt):
  txt = txt.lower()
  txt = re.sub(r'<.*?>',' ',txt)
  txt = re.sub(r'[^a-zA-Z]',' ',txt)
  txt = nltk.word_tokenize(txt)
  txt = [word for word in txt if word not in stopwords]
  txt = [word for word in txt if len(word)>3]
  stemming = PorterStemmer()
  txt = [stemming.stem(word) for word in txt]
  txt = ' '.join(txt)

  return txt



def get_keywords(docs,topN=10):
  docs_words_count = tfidf_transformer.transform(cv.transform([docs]))

  docs_words_count = docs_words_count.tocoo()
  tuple = zip(docs_words_count.col, docs_words_count.data)
  sorted_items = sorted(tuple, key = lambda x:(x[1],x[0]),reverse=True)

  sorted_items = sorted_items[:topN]

  score_vals = []
  feature_vals = []

  for idx, score in sorted_items:
    score_vals.append(round(score,3))
    feature_vals.append(feature_names[idx])

  result = {}
  for idx in range(len(feature_vals)):
      result[feature_vals[idx]] = score_vals[idx]

  return result






@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST','GET'])
def extract_keywords():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html',error='no file selected')
    if file:
        file = file.read().decode('utf-8',errors='ignore')
        cleaned_file = preprocessing_text(file)
        keywords = get_keywords(cleaned_file,20)

        return render_template('keywords.html',keywords=keywords)
    return render_template('index.html')




@app.route('/search_keywords', methods=['POST','GET'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = []

        for keyword in feature_names:
            if search_query.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keyword) == 20:
                    break

        print(keywords)

        return render_template('keywordslist.html',keywords=keywords)
    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)