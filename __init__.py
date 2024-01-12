from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import current_user, login_user, logout_user, login_required
from flask_login import LoginManager
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


app.config['SECRET_KEY'] = 'O4BSC0ONYNKAFJ9M'  # Remplacez par votre propre clé secrète


app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)

from app import routes, models

from flask_cors import CORS
