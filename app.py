from flask import Flask, jsonify, redirect, render_template, request, url_for
import numpy as np
from keras.preprocessing import image
import keras
import os
from keras.models import load_model
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

lemmatizer = WordNetLemmatizer()
model = load_model('model-development/chatbot_model.h5')
intents = json.loads(open('model-development/intents.json').read())
words = pickle.load(open('model-development/words.pkl','rb'))
classes = pickle.load(open('model-development/classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET', 'POST'])
def main():
    target = os.path.join(app_root, 'static/img/')
    if not os.path.isdir(target):
        os.makedirs(target)
        
    if request.method == 'GET':
        return render_template("index.html")

    elif request.method == 'POST':
        result = ""
        file = request.files['file']
        file_name = file.filename
        dest = '/'.join([target, file_name])
        file.save(dest)
        model = keras.models.load_model('model-development/model_2.h5')

        
        img = image.load_img(dest, target_size=(200, 200))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(classes[0])
        if classes[0]<0.5:
            result = "Kondisi tanah anda sudah cukup baik!"
        else:
            result = "Tanah anda kurang subur"
        return render_template("index.html" , result=result, filename='img/'+file_name)

    else:
        return "Unsupported Request Method"


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = chatbot_response(the_question)

    return jsonify({"response": response })


if __name__ == '__main__':
    app.run(debug=True, port=33507)