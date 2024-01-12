from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from app import db  # Assurez-vous que 'app' est bien défini et importé correctement

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True)
    lastname = db.Column(db.String(64))
    sexe = db.Column(db.String(10))
    pseudo = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(256))

    def __repr__(self):
        return f'<User {self.username}>'

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Création d'une clé étrangère pour la relation avec User
    ticker = db.Column(db.String(64))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def __repr__(self):
        return f'<Log {self.ticker} - {self.timestamp}>'
    


