from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,BooleanField,SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField('username',validators=[DataRequired(message='Input username...')])
    password = PasswordField('password',validators=[DataRequired(message='Input password...')])
    remember_me = BooleanField('Remember me')
    submit = SubmitField('Log in')


class DataForm(FlaskForm):
    data1 = StringField('data1',validators=[DataRequired(message='Input data1...')])
    data2 = StringField('data2',validators=[DataRequired(message='Input data2...')])
    data3 = StringField('data3', validators=[DataRequired(message='Input data3...')])
    submit = SubmitField('Evaluate')
