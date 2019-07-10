from flask import Flask
from flask import render_template,flash,redirect,send_file,request,url_for
from forms import LoginForm, DataForm
from werkzeug.utils import secure_filename
from config import Config
import tty
import webbrowser
import os


app = Flask(__name__)
app.config.from_object(Config)
address = 'x.csv'


@app.route('/')
@app.route('/index')
def index():
    user = {'username':'user'}
    return render_template('index.html',user=user)


@app.route('/update', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and tty.allowed_file(file.filename,app.config['ALLOWED_EXTENSIONS']):
            filename = secure_filename(file.filename)
            global address
            address=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('update.html')


@app.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('usernameis:{},remember it:{}'.format(
            form.username.data,form.remember_me.data))
        return redirect('/index')
    return render_template('login.html',title='Log in',form=form)


@app.route('/evaluator',methods=['GET','POST'])
def evaluator():
    form= DataForm()
    if form.validate_on_submit():
        data1=form.data1.data
        data2=form.data2.data
        data3=form.data3.data
        # evaluate here
        global address
        tty.run(data1,data2,address,data3)
        return redirect('/limeresult')
    return render_template('evaluator.html',title='Evaluator',form=form)


@app.route('/tensorflowresult')
def result():
    return render_template('result.html', title='Result')


@app.route('/limeresult')
def limeresult():
    return send_file("static/limeresult.html")



