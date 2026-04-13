import os
from flask import Flask, redirect, render_template, request, jsonify, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, FileField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired, Email, EqualTo, Length
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from CNN import CNN
# Try to import torch-related modules
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Disease prediction will not work.")
    TORCH_AVAILABLE = False
    # Define dummy classes/functions if torch is not available
    class torch:
        @staticmethod
        def no_grad():
            return lambda func: func
        class Tensor:
            def unsqueeze(self, dim):
                return self
        @staticmethod
        def argmax(tensor, dim=1):
            return type('obj', (object,), {'item': lambda self: 0})()

    class nn:
        class Module:
            def __init__(self):
                pass
            def eval(self):
                pass

    class transforms:
        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms
            def __call__(self, img):
                return img
        class Resize:
            def __init__(self, size):
                pass
        class ToTensor:
            def __init__(self):
                pass
import pandas as pd

# -----------------------------
# Load CSV Files
# -----------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# -----------------------------
# Temporary Model Definition
# -----------------------------
class TempModel(nn.Module):
    def __init__(self, num_classes=39):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))
        self.fc = nn.Linear(5 * 222 * 222, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------------
# Load the Model
# -----------------------------
model_path = "trained_model.pth"

if TORCH_AVAILABLE:
    model = CNN(K=len(disease_info))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model = checkpoint
        print("Model loaded successfully.")
    except Exception as e:
        print("Model load error:", e)
        model = None
else:
    model = None
    print("PyTorch not available - model prediction disabled")

if model:
    model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_path):
    if not TORCH_AVAILABLE or not model:
        return {'prediction': 0, 'confidence': 0.5, 'severity': 'moderate'}

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        pred = torch.argmax(output, dim=1).item()

    # Determine severity based on confidence and disease type
    severity = get_severity_level(pred, confidence)

    return {
        'prediction': pred,
        'confidence': confidence,
        'severity': severity
    }

# -----------------------------
# Severity Classification Function
# -----------------------------
def get_severity_level(disease_index, confidence):
    """
    Determine severity level based on disease type and confidence score
    """
    # High-risk diseases that should be treated seriously
    high_risk_diseases = ['Bacterial spot', 'Early blight', 'Late blight', 'Leaf curl', 'Mosaic virus']

    disease_name = disease_info.loc[disease_index, 'disease_name'] if disease_index < len(disease_info) else 'Unknown'

    # If confidence is low, might be early stage (mild)
    if confidence < 0.6:
        return 'mild'
    # If confidence is high and it's a high-risk disease, severe
    elif confidence > 0.85 and any(risk in disease_name for risk in high_risk_diseases):
        return 'severe'
    # Default to moderate
    else:
        return 'moderate'

# -----------------------------
# Flask App Configuration
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -----------------------------
# User Model
# -----------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    saved_supplements = db.relationship('SavedSupplement', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# -----------------------------
# Prediction Model
# -----------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    disease_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    prevent = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500))
    supplement_name = db.Column(db.String(200))
    supplement_image = db.Column(db.String(500))
    supplement_buy_link = db.Column(db.String(500))
    severity_level = db.Column(db.String(20), default='moderate')  # mild, moderate, severe
    confidence_score = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Relationships
    treatment_progress = db.relationship('TreatmentProgress', backref='prediction', lazy=True, cascade='all, delete-orphan')

# -----------------------------
# Saved Supplement Model
# -----------------------------
class SavedSupplement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    supplement_name = db.Column(db.String(200), nullable=False)
    supplement_image = db.Column(db.String(500))
    buy_link = db.Column(db.String(500))
    disease_name = db.Column(db.String(200))
    saved_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Ensure unique combination of user and supplement
    __table_args__ = (db.UniqueConstraint('user_id', 'supplement_name', name='unique_user_supplement'),)

# -----------------------------
# Treatment Progress Model
# -----------------------------
class TreatmentProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=False)
    progress_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    notes = db.Column(db.Text, nullable=False)
    improvement_status = db.Column(db.String(50), nullable=False)  # worse, same, improving, recovered
    follow_up_image = db.Column(db.String(500))  # Optional follow-up image path

class RemedySuggestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref='remedy_suggestions')

class ForumTopic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref='forum_topics')
    replies = db.relationship('ForumReply', backref='topic', lazy=True, cascade='all, delete-orphan')

class ForumReply(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('forum_topic.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref='forum_replies')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -----------------------------
# Forms
# -----------------------------
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=150)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class TreatmentProgressForm(FlaskForm):
    notes = StringField('Progress Notes', validators=[DataRequired()], widget=TextArea())
    improvement_status = SelectField('Improvement Status', choices=[
        ('worse', 'Getting Worse'),
        ('same', 'No Change'),
        ('improving', 'Improving'),
        ('recovered', 'Fully Recovered')
    ], validators=[DataRequired()])
    follow_up_image = FileField('Follow-up Image (Optional)')
    submit = SubmitField('Add Progress Update')

class RemedySuggestionForm(FlaskForm):
    title = StringField('Remedy Title', validators=[DataRequired(), Length(min=5, max=200)])
    description = StringField('Remedy Description', validators=[DataRequired(), Length(min=10)], widget=TextArea())
    submit = SubmitField('Submit Remedy')

class ForumTopicForm(FlaskForm):
    title = StringField('Question Title', validators=[DataRequired(), Length(min=10, max=200)])
    body = StringField('Question Details', validators=[DataRequired(), Length(min=20)], widget=TextArea())
    submit = SubmitField('Post Question')

class ForumReplyForm(FlaskForm):
    body = StringField('Your Reply', validators=[DataRequired(), Length(min=5)], widget=TextArea())
    submit = SubmitField('Post Reply')

# -----------------------------
# Authentication Routes
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home_page'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))

    form = SignupForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            flash('Email already exists. Please choose a different one.', 'danger')
        else:
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Your account has been created! You can now log in', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html', title='Sign Up', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home_page'))

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/alerts')
def alerts():
    regional_alerts = [
        {
            'region': 'North-Western Plains',
            'crop': 'Tomato',
            'disease': 'Early Blight',
            'advice': 'Inspect leaves daily and apply protective fungicide as needed.'
        },
        {
            'region': 'Central Valley',
            'crop': 'Potato',
            'disease': 'Late Blight',
            'advice': 'Avoid overhead irrigation and remove infected plants immediately.'
        },
        {
            'region': 'Eastern Highlands',
            'crop': 'Chili',
            'disease': 'Bacterial Spot',
            'advice': 'Ensure good airflow and use copper-based sprays when symptoms appear.'
        }
    ]
    return render_template('alerts.html', alerts=regional_alerts)

@app.route('/forum')
def forum():
    topics = ForumTopic.query.order_by(ForumTopic.created_at.desc()).all()
    return render_template('forum.html', topics=topics)

@app.route('/forum/new', methods=['GET', 'POST'])
@login_required
def new_forum_topic():
    form = ForumTopicForm()
    if form.validate_on_submit():
        topic = ForumTopic(
            user_id=current_user.id,
            title=form.title.data,
            body=form.body.data
        )
        db.session.add(topic)
        db.session.commit()
        flash('Your question has been posted to the forum.', 'success')
        return redirect(url_for('forum'))
    return render_template('new_forum_topic.html', form=form)

@app.route('/forum/topic/<int:topic_id>', methods=['GET', 'POST'])
def forum_topic(topic_id):
    topic = ForumTopic.query.get_or_404(topic_id)
    form = ForumReplyForm()
    if form.validate_on_submit():
        if not current_user.is_authenticated:
            flash('Please sign in to post a reply.', 'danger')
            return redirect(url_for('login'))
        reply = ForumReply(
            topic_id=topic_id,
            user_id=current_user.id,
            body=form.body.data
        )
        db.session.add(reply)
        db.session.commit()
        flash('Your reply has been posted.', 'success')
        return redirect(url_for('forum_topic', topic_id=topic_id))
    replies = ForumReply.query.filter_by(topic_id=topic_id).order_by(ForumReply.created_at.asc()).all()
    return render_template('forum_topic.html', topic=topic, replies=replies, form=form)

@app.route('/experts')
def experts():
    expert_list = [
        {'name': 'Dr. Kavita Sharma', 'specialty': 'Plant Pathology', 'email': 'ksharma@plantcure.org'},
        {'name': 'Mr. Arjun Patel', 'specialty': 'Sustainable Farming', 'email': 'arjun.patel@plantcure.org'},
        {'name': 'Ms. Nisha Rao', 'specialty': 'Crop Protection', 'email': 'nisha.rao@plantcure.org'}
    ]
    return render_template('experts.html', experts=expert_list)

@app.route('/schemes')
def schemes():
    schemes_info = [
        {'name': 'National Agriculture Insurance Scheme', 'description': 'Insurance support for crop losses and damage caused by pests and disease.'},
        {'name': 'Subsidized Inputs Program', 'description': 'Financial support for approved fertilizers, seeds, and crop protection products.'},
        {'name': 'Farmer Training Grant', 'description': 'Skill-building and advisory programs for modern disease management techniques.'}
    ]
    return render_template('schemes.html', schemes=schemes_info)

@app.route('/submit-remedy', methods=['GET', 'POST'])
@login_required
def submit_remedy():
    form = RemedySuggestionForm()
    suggestions = RemedySuggestion.query.order_by(RemedySuggestion.submitted_at.desc()).all()

    if form.validate_on_submit():
        suggestion = RemedySuggestion(
            user_id=current_user.id,
            title=form.title.data,
            description=form.description.data
        )
        db.session.add(suggestion)
        db.session.commit()
        flash('Thank you! Your remedy has been shared with the community.', 'success')
        return redirect(url_for('submit_remedy'))

    return render_template('submit_remedy.html', form=form, suggestions=suggestions)

@app.route('/index')
@login_required
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
@login_required
def submit():
    if request.method == 'POST':

        upload_path = "static/uploads"
        os.makedirs(upload_path, exist_ok=True)

        image = request.files['image']
        file_path = os.path.join(upload_path, image.filename)
        image.save(file_path)

        # Get prediction results
        prediction_result = predict(file_path)

        if isinstance(prediction_result, dict):
            pred = prediction_result['prediction']
            confidence = prediction_result['confidence']
            severity = prediction_result['severity']
        else:
            # Fallback for old format
            pred = prediction_result
            confidence = 0.5
            severity = 'moderate'

        title = disease_info.loc[pred, 'disease_name']
        description = disease_info.loc[pred, 'description']
        prevent = disease_info.loc[pred, 'Possible Steps']
        image_url = disease_info.loc[pred, 'image_url']

        supplement_name = supplement_info.loc[pred, 'supplement name']
        supplement_image = supplement_info.loc[pred, 'supplement image']
        supplement_buy_link = supplement_info.loc[pred, 'buy link']

        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            image_path=file_path,
            disease_name=title,
            description=description,
            prevent=prevent,
            image_url=image_url,
            supplement_name=supplement_name,
            supplement_image=supplement_image,
            supplement_buy_link=supplement_buy_link,
            severity_level=severity,
            confidence_score=confidence
        )
        db.session.add(prediction)
        db.session.commit()

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image,
            buy_link=supplement_buy_link,
            severity=severity,
            confidence=f"{confidence:.2%}",
            prediction_id=prediction.id
        )

    return redirect('/')

@app.route('/market')
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# -----------------------------
# New Feature Routes
# -----------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('dashboard.html', predictions=predictions)

@app.route('/saved-remedies')
@login_required
def saved_remedies():
    """Display user's saved/bookmarked supplements"""
    saved_supplements = SavedSupplement.query.filter_by(user_id=current_user.id).order_by(SavedSupplement.saved_at.desc()).all()
    return render_template('saved_remedies.html', saved_supplements=saved_supplements)

@app.route('/bookmark-supplement/<supplement_name>', methods=['POST'])
@login_required
def bookmark_supplement(supplement_name):
    """Bookmark a supplement"""
    # Find supplement details from the CSV
    supplement_row = supplement_info[supplement_info['supplement name'] == supplement_name]
    if not supplement_row.empty:
        row = supplement_row.iloc[0]
        disease_name = request.form.get('disease_name', '')

        # Check if already bookmarked
        existing = SavedSupplement.query.filter_by(
            user_id=current_user.id,
            supplement_name=supplement_name
        ).first()

        if not existing:
            saved_supplement = SavedSupplement(
                user_id=current_user.id,
                supplement_name=supplement_name,
                supplement_image=row['supplement image'],
                buy_link=row['buy link'],
                disease_name=disease_name
            )
            db.session.add(saved_supplement)
            db.session.commit()
            flash(f'{supplement_name} has been saved to your remedies!', 'success')
        else:
            flash(f'{supplement_name} is already in your saved remedies.', 'info')
    else:
        flash('Supplement not found.', 'error')

    return redirect(request.referrer or url_for('market'))

@app.route('/remove-bookmark/<int:bookmark_id>', methods=['POST'])
@login_required
def remove_bookmark(bookmark_id):
    """Remove a bookmarked supplement"""
    bookmark = SavedSupplement.query.filter_by(id=bookmark_id, user_id=current_user.id).first()
    if bookmark:
        db.session.delete(bookmark)
        db.session.commit()
        flash(f'{bookmark.supplement_name} has been removed from your saved remedies.', 'success')
    else:
        flash('Bookmark not found.', 'error')

    return redirect(request.referrer or url_for('saved_remedies'))

@app.route('/prediction/<int:prediction_id>')
@login_required
def prediction_detail(prediction_id):
    """View detailed prediction with treatment progress"""
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first()
    if not prediction:
        flash('Prediction not found.', 'error')
        return redirect(url_for('dashboard'))

    treatment_progress = TreatmentProgress.query.filter_by(prediction_id=prediction_id).order_by(TreatmentProgress.progress_date.desc()).all()

    return render_template('prediction_detail.html',
                         prediction=prediction,
                         treatment_progress=treatment_progress)

@app.route('/add-progress/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def add_treatment_progress(prediction_id):
    """Add treatment progress update"""
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first()
    if not prediction:
        flash('Prediction not found.', 'error')
        return redirect(url_for('dashboard'))

    form = TreatmentProgressForm()
    if form.validate_on_submit():
        # Handle follow-up image upload
        follow_up_image_path = None
        if form.follow_up_image.data:
            upload_path = "static/uploads/progress"
            os.makedirs(upload_path, exist_ok=True)
            image = form.follow_up_image.data
            filename = f"progress_{prediction_id}_{len(prediction.treatment_progress)}_{image.filename}"
            file_path = os.path.join(upload_path, filename)
            image.save(file_path)
            follow_up_image_path = file_path

        # Add progress entry
        progress = TreatmentProgress(
            prediction_id=prediction_id,
            notes=form.notes.data,
            improvement_status=form.improvement_status.data,
            follow_up_image=follow_up_image_path
        )
        db.session.add(progress)
        db.session.commit()

        flash('Treatment progress has been recorded!', 'success')
        return redirect(url_for('prediction_detail', prediction_id=prediction_id))

    return render_template('add_progress.html', form=form, prediction=prediction)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
