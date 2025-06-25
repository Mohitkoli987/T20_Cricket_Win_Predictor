from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cricket.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to store uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    batting_team = db.Column(db.String(50), nullable=False)
    bowling_team = db.Column(db.String(50), nullable=False)
    runs = db.Column(db.Integer, nullable=False)
    wickets = db.Column(db.Integer, nullable=False)
    overs = db.Column(db.Float, nullable=False)
    predicted_score = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Sample Articles Data
def seed_articles():
    articles = [
        {
            'title': 'The Evolution of T20 Cricket',
            'content': '''T20 cricket has revolutionized the sport since its inception in 2003. The format has brought new 
            strategies, innovative shots, and a fresh approach to the game. Teams now focus on maintaining high strike rates 
            while also managing risks effectively. The IPL has been at the forefront of this evolution, showcasing some of 
            the most exciting cricket matches and introducing new tactical elements to the sport.''',
            'image_url': 'https://images.unsplash.com/photo-1531415074968-036ba1b575da?ixlib=rb-4.0.3'
        },
        {
            'title': 'Understanding IPL Analytics',
            'content': '''Cricket analytics has become increasingly sophisticated in the IPL. Teams use data science to analyze 
            player performance, predict match outcomes, and make strategic decisions. From ball-by-ball analysis to player 
            match-ups, the role of data in cricket has never been more important. This article explores how teams use analytics 
            to gain a competitive edge in the IPL.''',
            'image_url': 'https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?ixlib=rb-4.0.3'
        },
        
    ]

    for article_data in articles:
        article = Article.query.filter_by(title=article_data['title']).first()
        if not article:
            article = Article(**article_data)
            db.session.add(article)
    
    db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    articles = Article.query.order_by(Article.created_at.desc()).limit(6).all()
    return render_template('index.html', articles=articles)

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for adding articles
@app.route('/add_article', methods=['GET', 'POST'])
def add_article():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        image = request.files.get('image')

        if not title or not content:
            flash('Title and content are required!', 'danger')
            return redirect(url_for('add_article'))

        image_url = None
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image_url = f"/{app.config['UPLOAD_FOLDER']}/{filename}"

        try:
            new_article = Article(
                title=title,
                content=content,
                image_url=image_url,
                created_at=datetime.utcnow()
            )
            db.session.add(new_article)
            db.session.commit()
            flash('Article added successfully!', 'success')
            return redirect(url_for('articles'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding article: {str(e)}', 'danger')
            return redirect(url_for('add_article'))

    return render_template('add_article.html')

# Ensure database is created
with app.app_context():
    db.create_all()

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        selected_city = request.form['selected_city']
        target = int(request.form['target'])
        score = int(request.form['score'])
        balls_left = int(request.form['balls_left'])
        wickets = int(request.form['wickets'])

        runs_left = target - score
        wickets_remaining = 10 - wickets
        overs_completed = (120 - balls_left) / 6
        crr = score / overs_completed if overs_completed > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')

        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_remaining': [wickets_remaining],
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        pipe = pickle.load(open('ra_pipe.pkl', 'rb'))
        result = pipe.predict_proba(input_data)

        win_probability = round(result[0][1] * 100)
        loss_probability = round(result[0][0] * 100)

        # Save prediction to history
        prediction = Prediction(
            user_id=current_user.id,
            batting_team=batting_team,
            bowling_team=bowling_team,
            runs=score,
            wickets=wickets,
            overs=overs_completed,
            predicted_score=win_probability
        )
        db.session.add(prediction)
        db.session.commit()

        return render_template('result.html', 
                             batting_team=batting_team, 
                             bowling_team=bowling_team, 
                             win_probability=win_probability, 
                             loss_probability=loss_probability)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        user = User(username=username, 
                   email=email, 
                   password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('profile'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('profile.html', predictions=predictions)

@app.route('/articles')
def articles():
    articles = Article.query.order_by(Article.created_at.desc()).all()
    return render_template('articles.html', articles=articles)

@app.route('/article/<int:id>')
def article(id):
    article = Article.query.get_or_404(id)
    return render_template('article.html', article=article)

# Initialize database and seed articles
with app.app_context():
    db.create_all()
    seed_articles()

if __name__ == '__main__':
    app.run(debug=True)
