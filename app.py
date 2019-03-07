from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

"""Try dropdown box"""
@app.route('/dropdown', methods=['GET'])
def dropdown():
    colours = ['Red', 'Blue', 'Black', 'Orange']
    return render_template('dropdown.html', colours=colours)

"""END"""

@app.route('/predict',methods=['POST'])
def predict():
	
	# with open('df_2012b_clean.pkl', "rb") as f:
	# 	df_2012b = pickle.load(f)

	# # Features and Labels
	# df_root = df_2012b
	# X = df_root.text_clean
	# y = df_root.positive.values
	
	# Extract Feature With CountVectorizer
	# from nltk.corpus import stopwords
	# stop = stopwords.words('english')
	# # add some stop words
	# stop += ['br', 'br br', 'http', 'www', 'amazon', 'com', 'href', 'br /', '<br />', 'gp']
	
	# # define min_df & max_df
	# min_df = 10
	# max_df = 0.9

	# cv1 = CountVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df, max_features=1000,
 #                     binary=True, stop_words=stop) 
	# cvect = cv1 
	
	# from sklearn.model_selection import train_test_split
	# X_train, X_test, y_train, y_test = train_test_split(
	# X, y, test_size=0.20, stratify=y, random_state=0)#Naive Bayes Classifier
	# X_train_cvect = cvect.fit_transform(X_train)
	# X_test_cvect = cvect.transform(X_test)

	# from imblearn.ensemble import BalancedRandomForestClassifier

	# brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
	# brf.fit(X_train_cvect, y_train)
	# brf.score(X_test_cvect,y_test)

	# Alternative Usage of Saved Model
	brf_model = open('/Users/darrenchia/Desktop/Flask/sentiment/brf_model.pkl','rb')
	brf = joblib.load(brf_model)

	brf_cvect = open('/Users/darrenchia/Desktop/Flask/sentiment/cvect.pkl','rb')
	cvect = joblib.load(brf_cvect)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cvect.transform(data).toarray()
		my_prediction = brf.predict(vect)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run()
	# app.run(debug=True)