from logging import NullHandler
from sys import maxsize
from flask import Flask, render_template, request, url_for, redirect
from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline

mode_name1 = 'liam168/trans-opus-mt-zh-en'
model1 = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer1 = AutoTokenizer.from_pretrained(mode_name)
translation1 = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer )





app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        scoure = request.form['scoure']
        login_massage = scoure
        message = translation(scoure, max_length=400)[0]
        next_message = message['translation_text']
        print(login_massage, message, next_message)
        if login_massage.strip()=='':
            next_message = ''
            return render_template('demo', message = login_massage, message1 = next_message)
    return render_template('demo')


if __name__=='__main__':
   app.run(host='0.0.0.0', port=8080,debug=True)