from app import app, db
from app.forms import LoginForm
from app.models import User, Log
from flask import jsonify, request, flash, render_template, redirect, url_for
from flask_login import current_user
from sqlalchemy.exc import IntegrityError
from keras.models import load_model
from PIL import Image
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
import base64
import io  # Pas besoin d'importer BytesIO séparément si nous avons déjà importé io
from datetime import datetime  # Importation en double, nous avons gardé la première
from io import BytesIO
import os
import re
from werkzeug.utils import secure_filename

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        lastname = request.form['lastname']
        sexe = request.form['sexe']
        pseudo = request.form['pseudo']
        email = request.form['email']

        # Vérifiez si l'utilisateur existe déjà
        user_exist = User.query.filter((User.pseudo == pseudo) | (User.email == email)).first()
        if user_exist:
            message = "Erreur : cet email ou pseudo existe déjà. Veuillez essayer avec des valeurs différentes."
            return render_template('register.html', title='register', message=message)

        new_user = User(username=username, lastname=lastname, sexe=sexe, pseudo=pseudo, email=email)

        try:
            db.session.add(new_user)
            db.session.commit()
            message = f"Bonjour {username}, votre compte a été créé avec succès !"
        except IntegrityError:
            db.session.rollback()
            message = "Une erreur inattendue s'est produite. Veuillez réessayer."

        return render_template('register.html', title='register', message=message)

    return render_template('register.html', title='register')



@app.route('/utilisateurs_inscrits', methods=['GET', 'POST'])
def utilisateurs_inscrits():
    # Récupérer tous les utilisateurs pour les afficher
    users = User.query.all()
    print(users)
    return render_template('utilisateurs_inscrits.html', users=users)

@app.route('/formulaire_recherches', methods=['GET', 'POST'])
def formulaire_recherches():
    if request.method == 'POST':
        ticker = request.form['ticker']
        news = obtenir_news(ticker)  # Fonction pour obtenir les nouvelles

        if news:
            if current_user.is_authenticated:
                user_id = current_user.id  # Obtention de l'ID de l'utilisateur connecté
                enregistrer_log(ticker, user_id)  # Enregistrement du log
            else:
                flash("Vous devez être connecté pour effectuer cette action.")
                return redirect(url_for('login'))  # Redirection vers la page de connexion
            
            return render_template('afficher_news.html', news=news)
        else:
            flash("Aucune nouvelle trouvée pour le symbole boursier donné.")

    return render_template('formulaire_recherche.html')

from app import login_manager
from .models import User  # Importez votre modèle User

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



@app.route('/recherche_news')
def recherche_news():
    ticker = request.args.get('ticker')
    if ticker:
        news = obtenir_news(ticker)

        # Enregistrer le log si l'utilisateur est connecté
        if current_user.is_authenticated:
            user_id = current_user.id
            enregistrer_log(ticker, user_id)

        return jsonify(news)
    return jsonify({"error": "Aucun ticker fourni"})

def enregistrer_log(ticker, user_id):
    nouveau_log = Log(ticker=ticker, user_id=user_id, timestamp=datetime.now())
    db.session.add(nouveau_log)
    db.session.commit()

def obtenir_news(ticker):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",  # Exemple de fonction, choisissez en fonction de vos besoins
        "symbol": ticker,
        "apikey": "O4BSC0ONYNKAFJ9M"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des informations : {e}")
        return None
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des nouvelles : {e}")
        return None

    # Traitement des données reçues
    if response:
        time_series = response.json().get("Time Series (Daily)", {})
        # Structurez les données comme vous le souhaitez, par exemple :
        structured_data = []
        for date, data in time_series.items():
            structured_data.append({
                "date": date,
                "open": data["1. open"],
                "high": data["2. high"],
                "low": data["3. low"],
                "close": data["4. close"],
                "volume": data["5. volume"]
            })
        return structured_data
    return None

@app.route('/afficher_logs')
def afficher_logs():
    logs = Log.query.all()  # Remplacez Log par votre modèle de log
    return render_template('afficher_logs.html', logs=logs)

@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        f = request.files['datafile']
        if f.filename.endswith('.csv'):
            df = pd.read_csv(f)
        elif f.filename.endswith('.xlsx'):
            df = pd.read_excel(f)
        else:
            return 'Format de fichier non pris en charge', 400

        stats = df.describe().to_html()
        return render_template('data_stats.html', tables=[stats])
    return render_template('upload_data.html')


@app.route('/stock_chart', methods=['GET', 'POST'])
def stock_chart():
    # Initialisation des variables
    graph_url = None
    symbol = None

    if request.method == 'POST':
        # Récupérer le symbole boursier du formulaire
        symbol = request.form.get('ticker')
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": "O4BSC0ONYNKAFJ9M"
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)

        if response.status_code == 200:
            data = response.json()
            time_series = data['Time Series (Daily)']
            dates = []
            prices = []
            for date_str, price_data in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date2num(date))  # Convertit les dates en format numérique pour matplotlib
                prices.append(float(price_data['4. close']))

            # Création et sauvegarde du graphique
            plt.figure(figsize=(10, 5))
            plt.plot_date(dates, prices, '-')
            plt.title('Graphique du Prix des Actions ' + symbol)
            plt.xlabel('Date')
            plt.ylabel('Prix de clôture')
            plt.gcf().autofmt_xdate()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            graph_url = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        else:
            flash(f"Erreur lors de la requête API : {response.status_code}", 'error')

    # Rendre le template même si c'est une requête GET, pour afficher le formulaire
    return render_template('stock_chart.html', graph_url=graph_url, symbol=symbol)

#################################
# Chemins et configuration
model_path = 'C:\\Users\\lemai\\doclouiskuhn\\IA-P3-Euskadi\\Projets\\Projet P3 - Flask\\exercice\\BDD\\app\\models\\handwritten.model'
UPLOAD_FOLDER = 'C:\\Users\\lemai\\doclouiskuhn\\IA-P3-Euskadi\\Projets\\Projet P3 - Flask\\exercice\\BDD\\app\\digits'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Chargement du modèle
model = load_model(model_path)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
######################################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
######################################
def prepare_image(image):
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image /= 255
    return image

########################
def make_prediction(image):
    image_np = prepare_image(image)
    prediction = model.predict(image_np).argmax()
    return prediction

@app.route('/mnist_upload', methods=['GET', 'POST'])
def mnist_upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('mnist_upload.html', error='Aucun fichier n\'a été téléchargé ou sélectionné.')
    elif file:
        image = Image.open(file.stream)
        prepared_image = prepare_image(image)
        prediction = model.predict(prepared_image).argmax()
        return f'Prédiction du Modèle: {prediction}'
    else:
         return render_template('mnist_upload.html')
#####################################################################
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        image_b64 = data['image']

        # Conversion de l'image base64 en un objet image
        image_bytes = io.BytesIO(base64.b64decode(image_b64))
        image = Image.open(image_bytes).convert('L')  # Convertir en niveaux de gris
        image = image.resize((28, 28), Image.Resampling.LANCZOS)  # Redimensionner pour le modèle

        # Préparation de l'image pour la prédiction
        image_np = np.array(image).reshape(1, 28, 28, 1)
        image_np = image_np.astype('float32') / 255.0

        # Faire la prédiction en utilisant le modèle chargé
        prediction = model.predict(image_np).argmax()

        return jsonify({'prediction': int(prediction)})

    # Pour les requêtes GET, affichez simplement la page
    return render_template('predict.html')




