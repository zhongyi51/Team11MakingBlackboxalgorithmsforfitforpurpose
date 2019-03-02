from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,BooleanField,SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField('username',validators=[DataRequired(message='Input username...')])
    password = PasswordField('password',validators=[DataRequired(message='Input password...')])
    remember_me = BooleanField('Remember me')
    submit = SubmitField('Log in')


class DataForm(FlaskForm):
    data1 = StringField('Categorical Labels(exclude target)',validators=[DataRequired(message='Input data1...')])
    data2 = StringField('Target Label',validators=[DataRequired(message='Input data2...')])
    data3 = StringField('Predictable Sample', validators=[DataRequired(message='Input data3...')])
    submit = SubmitField('Evaluate')
