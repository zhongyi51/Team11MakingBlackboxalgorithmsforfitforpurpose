import os
class Config(object):
    SECRET_KEY = '123456'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__),'file')
    ALLOWED_EXTENSIONS = set(['csv'])
