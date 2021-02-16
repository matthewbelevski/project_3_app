from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'real_or_fake_model.pkl'
#Loading Naive Bayes model
news_model = pickle.load(open(filename, 'rb'))
#count vectorizer
cv = pickle.load(open('vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = [str(x) for x in request.form.values()][0]
		data = [message]
        #data = cv.transform([message])
        #vect = cv(data).toarray()
		my_prediction = news_model.predict(cv.transform(data))
		if(my_prediction[0] == 1):
    			output = "static/real_news_trump.jpg"
		elif(my_prediction[0] == 0):
    			output = "static/fake_news_trump.jpg"
	return render_template('index.html',prediction_text = output)



if __name__ == '__main__':
	app.run(debug=True)
