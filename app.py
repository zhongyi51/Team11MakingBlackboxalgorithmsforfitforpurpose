from flask import Flask
from flask import render_template,flash,redirect
from forms import LoginForm, DataForm
from config import Config
app = Flask(__name__)
app.config.from_object(Config)


@app.route('/')
@app.route('/index')
def index():
    user = {'username':'user'}
    return render_template('index.html',user=user)


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
        # evaluate here
        return redirect('/result')
    return render_template('evaluator.html',title='Evaluator',form=form)


@app.route('/result')
def result():
    return render_template('result.html', title='Result')
