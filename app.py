### Integrate HTML With Flask
### HTTP verb GET And POST
from flask import Flask,redirect,url_for,render_template,request,make_response
import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import seaborn as sb
from wordcloud import WordCloud
from collections import Counter
from keras.utils import pad_sequences
import tensorflow as tf
tokenizer = tf.keras.preprocessing.text.Tokenizer()
from io import BytesIO
import base64
from wordcloud import WordCloud
from collections import Counter
import os
import matplotlib
import tensorflow as tf
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import re
import pickle
matplotlib.use('Agg')
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
total_stopwords = set(stopwords.words('english'))
negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
final_stopwords = total_stopwords - negative_stop_words
final_stopwords.add("one")
stemmer = PorterStemmer()

# ---------------------------------------------
HTMLTAGS = re.compile('<.*?>')
table = str.maketrans(dict.fromkeys(string.punctuation))
remove_digits = str.maketrans('', '', string.digits)
MULTIPLE_WHITESPACE = re.compile(r"\s+")
def preprocessor(review):
    review = HTMLTAGS.sub(r'', review)
    review = review.translate(table)
    review = review.translate(remove_digits)
    review = review.lower()
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    review = [word for word in review.split()
              if word not in final_stopwords]
    review = ' '.join([stemmer.stem(word) for word in review])
    return review
import pickle



app=Flask(__name__)
picFolder = os.path.join('static','pics')
app.config['UPLOAD_FOLDER']=picFolder




#HOME
@app.route('/',methods=['GET','post'])
def home():
    food1=os.path.join(app.config['UPLOAD_FOLDER'],'bg2.jpg')
    food2=os.path.join(app.config['UPLOAD_FOLDER'],'bg3.png')
    food3=os.path.join(app.config['UPLOAD_FOLDER'],'bg4.jpg')
    food4=os.path.join(app.config['UPLOAD_FOLDER'],'sentiment_analysis.jpg')
    return render_template('index.html',image1= food1,image2=food2,image3=food3,image4=food4)
#About_Analysis
@app.route('/predict',methods=['GET','post'])
def predict():
    food=os.path.join(app.config['UPLOAD_FOLDER'],'image3.jpg')
    foods=os.path.join(app.config['UPLOAD_FOLDER'],'method.png') 
    return render_template('app_2.html',user_image =food,user_image1=foods)
#PREDICTION VADAR
@app.route('/second', methods=['GET', 'POST'])
def second():
    if request.method == 'POST':
        if 'text1' in request.form:
            input = request.form['text1']
            sia = SentimentIntensityAnalyzer()
            polarity = sia.polarity_scores(input)
            sentiment_pipeline = pipeline("sentiment-analysis")
            result = sentiment_pipeline(input)
            foods=os.path.join(app.config['UPLOAD_FOLDER'],'sentiment-analysis.jpg') 
                
            # Tokenize the text
            tokens = input.split()

            # Count word frequency
            freq = Counter(tokens)

            # Create a WordCloud object
            wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate_from_frequencies(freq)
             # Convert the Matplotlib plot into an image buffer
            img_buffer = io.BytesIO()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plot_url2 = base64.b64encode(img_buffer.getvalue()).decode()
            return render_template('second.html', text=input,polarity=polarity,plot_url2=plot_url2, prediction=result,image =foods)
    foods=os.path.join(app.config['UPLOAD_FOLDER'],'sentiment-analysis.jpg') 
    return render_template('second.html',image =foods)


#   CHART

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'csv_file' not in request.files:
            return 'No file uploaded.'
        
        # Read uploaded file as a pandas DataFrame
        csv_file = request.files['csv_file']
        df = pd.read_csv(csv_file)
        score_counts = df['Score'].value_counts().sort_index()
        score_counts.index = ["1 star", "2 star", "3 star", "4 star", "5 star"]
        total_reviews = score_counts.sum()
        score_counts = score_counts.append(pd.Series(total_reviews, index=['Total']))
        score_counts.index.name = 'Stars'

        data = pd.DataFrame(score_counts,columns=['Counts'])
        data = data.transpose()
        df_frame = data.to_html(index=False)
        plt.style.use('ggplot')
        textprops = {"fontsize":10}
        # Plot reviews count by score
        fig, ax = plt.subplots(figsize=(7,7))
        df['Score'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', shadow=True, textprops=textprops, ax=ax)
        ax.set_title('Count of Reviews by Stars', fontsize=14)
        ax.legend(labels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars'], loc='center left', bbox_to_anchor=(1.2, 0.5))
        plt.tight_layout()
        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()

        # Perform sentiment analysis on reviews
        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []
        for index, row in df.iterrows():
            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(row['Text'])['compound']
            if sentiment_score > 0.05:
                positive_reviews.append(row['Text'])
            elif sentiment_score < -0.05:
                negative_reviews.append(row['Text'])
            else:
                neutral_reviews.append(row['Text'])

        # Plot sentiment analysis
        x_axis = ['Positive', 'Negative', 'Neutral']
        y_axis =[len(positive_reviews), len(negative_reviews), len(neutral_reviews)]
        data_frame = pd.DataFrame({'Sentiment': x_axis, 'Number of Reviews': y_axis})
        df_frames = data_frame.to_html(index=False)
        sb.set_style("whitegrid")
        fig,axs = plt.subplots()
        sb.barplot(x=x_axis, y=y_axis)
        axs.bar_label(axs.containers[0])
        # Remove top and right borders

        axs.set_xlabel('Sentiment')
        axs.set_ylabel('Number of Reviews')
        axs.set_title('Sentiment Analysis')
        plt.tight_layout()
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        foods=os.path.join(app.config['UPLOAD_FOLDER'],'sentiment-analysis-positive-negative-mentions-charts.jpg')
        return render_template('app_3.html', plot_url1=plot_url1, plot_url2=plot_url2,image=foods,df=df_frame,data=df_frames)
    foods=os.path.join(app.config['UPLOAD_FOLDER'],'sentiment-analysis-positive-negative-mentions-charts.jpg')
    return render_template('app_3.html',image=foods)

@app.route('/lstm',methods=['GET','POST'])
def lstm():
    if request.method == 'POST':
        if 'text1' in request.form:
            review = request.form['text1']
            text = review
            review = HTMLTAGS.sub(r'', text)
            review = review.translate(table)
            review = review.translate(remove_digits)
            review = review.lower()
            review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
            review = [word for word in review.split() if word not in final_stopwords]
            review = ' '.join([stemmer.stem(word) for word in review])
            x = preprocessor(review)
            tfidf_vectorizer= pickle.load(open("transformer.pkl",'rb'))
            bmodel= pickle.load(open("model.pkl",'rb'))
            x = tfidf_vectorizer.transform([x])
            y = int(bmodel.predict(x.reshape(1,-1)))
            label=['Negative', 'Neutral', 'Positive']
            ans = label[y]
            foods=os.path.join(app.config['UPLOAD_FOLDER'],'bg5.png') 
            return render_template('app_4.html',text=text,prediction=ans,image= foods)
    foods=os.path.join(app.config['UPLOAD_FOLDER'],'bg5.png') 
    return render_template('app_4.html',image = foods)



from flask import Flask, render_template
import plotly.graph_objs as go
import requests
import re
import bs4
from bs4 import BeautifulSoup
import plotly.express as px



@app.route('/url',methods=['GET','POST'])
def url():
    if request.method == 'POST':
        if 'my-url' not in request.form:
            return 'No url entered.'
        url = request.form['my-url']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }


        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        rating = soup.find('span', {'class': 'a-icon-alt'})
        if rating:
            rating = rating.get_text().strip()
        else:
            rating = 'Rating not found'
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            product_title_element = soup.find('span', {'id': 'productTitle'})
            if product_title_element is not None:
                product_title = product_title_element.text.strip()
                product_names= product_title
            else:
                product_names = 'Product title not found'
    
        reviews = []

        for review in soup.find_all('div', {'data-hook': 'review'}):
            review_text = review.find('span', {'data-hook': 'review-body'}).text.strip()
            reviews.append((review_text))


        # create a DataFrame from the reviews
        reviews_df = pd.DataFrame(reviews, columns=['review'])

        # use SentimentIntensityAnalyzer to get the sentiment score for each review
        sia = SentimentIntensityAnalyzer()

        positive_reviews = []
        negative_reviews = []
        neutral_reviews = []

        for index, row in reviews_df.iterrows():
            sentiment_score = sia.polarity_scores(row['review'])['compound']
            if sentiment_score > 0.05:
                positive_reviews.append(row['review'])
            elif sentiment_score < -0.05:
                negative_reviews.append(row['review'])
            else:
                neutral_reviews.append(row['review'])

        

        x_axis = ['Positive', 'Negative', 'Neutral']
        y_axis = [len(positive_reviews), len(negative_reviews), len(neutral_reviews)]

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x_axis, y_axis, color='#7ed6df')

        # Adding labels and titles
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Sentiment Analysis')

        # Adding values to the bars
        for i in ax.containers:
            ax.bar_label(i, label_type='edge')
        plt.tight_layout()
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        
        # create a DataFrame with the data
        df = pd.DataFrame({'Sentiment': x_axis, 'Number of Reviews': y_axis})
        df_frame = df.to_html(index=False)
        # Get the top 2 positive and negative reviews
        top_positive_reviews = sorted(positive_reviews, key=lambda x: sia.polarity_scores(x)['compound'], reverse=True)[:2]
        top_negative_reviews = sorted(negative_reviews, key=lambda x: sia.polarity_scores(x)['compound'])[:2]
        foods=os.path.join(app.config['UPLOAD_FOLDER'],'logo_white.jpg') 
        return render_template('app_5.html',name= product_names,rating=rating ,df=df_frame,positive_reviews=top_positive_reviews,negative_reviews=top_negative_reviews,plot_url2=plot_url2,image=foods)
    foods=os.path.join(app.config['UPLOAD_FOLDER'],'logo_white.jpg')
    return render_template('app_5.html',image=foods)

if __name__ == "__main__":
   app.run(debug=True, use_reloader=False,host='169.254.246.227',port=5000)