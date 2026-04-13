# CultiKure - Plant Disease Detection System
## Technical Project Review Document

---

## 1. PROJECT OVERVIEW

**CultiKure** is a web-based agricultural intelligence platform designed to empower farmers and agricultural experts with AI-powered disease detection capabilities. The application combines machine learning inference with community-driven knowledge sharing to provide comprehensive plant health management solutions.

**Core Objective:** Enable farmers to quickly identify crop diseases, access treatment recommendations, track recovery progress, and connect with agricultural experts.

**Target Users:**
- Small-scale farmers
- Agricultural professionals
- Gardeners
- Agricultural researchers

---

## 2. TECHNOLOGY STACK

### Backend
- **Framework:** Flask 2.x (Python web framework)
- **Database:** SQLite with SQLAlchemy ORM
- **Authentication:** Flask-Login with Werkzeug password hashing
- **Form Validation:** Flask-WTF with WTForms
- **Image Processing:** Python Imaging Library (PIL)

### Machine Learning
- **Deep Learning Framework:** PyTorch
- **Model Architecture:** ResNet50 (ResNet50.pt)
- **Image Transformations:** torchvision.transforms
- **Input Shape:** 224×224 RGB images
- **Output Classes:** 39 disease categories

### Frontend
- **Template Engine:** Jinja2
- **CSS Framework:** Bootstrap 5.3.2
- **UI Styling:** Custom CSS with responsive design
- **JavaScript:** Vanilla JS for file upload handling
- **Fonts:** Google Fonts (Poppins)

### Database
- **Type:** SQLite
- **File:** users.db
- **ORM:** SQLAlchemy 1.4+

---

## 3. SYSTEM ARCHITECTURE

### High-Level Flow
```
User Upload Image
    ↓
Image Processing & Validation
    ↓
ResNet50 Model Inference
    ↓
Disease Prediction & Severity Classification
    ↓
Database Storage (Prediction Record)
    ↓
Display Results + Treatment Recommendations
```

### Core Components

#### 3.1 Authentication System
- User registration with username, email, and password validation
- Password hashing using Werkzeug security utilities
- Session management via Flask-Login
- Login persistence with "Remember Me" functionality

#### 3.2 Disease Detection Pipeline
- Image upload to `/submit` route
- Image validation and storage in `static/uploads/`
- ML inference through `predict()` function
- Confidence score calculation using softmax
- Severity classification based on:
  - Confidence threshold (<60% = mild, >85% with high-risk disease = severe)
  - High-risk disease list: Bacterial spot, Early blight, Late blight, Leaf curl, Mosaic virus

#### 3.3 Prediction Tracking
- Store prediction history per user
- Link predictions to treatment progress logs
- Display prediction timeline in dashboard

#### 3.4 Community Features
- **Forum System:** User-posted questions with threaded replies
- **Remedy Sharing:** Community-submitted treatment ideas
- **Regional Alerts:** Location-specific disease warnings
- **Expert Q&A:** Directory of agricultural specialists

---

## 4. DATABASE SCHEMA

### Models

#### User Model
```
- id (Primary Key)
- username (unique, 150 chars)
- email (unique, 150 chars)
- password_hash (150 chars)
- Relationships: predictions, saved_supplements, remedy_suggestions, forum_topics, forum_replies
```

#### Prediction Model
```
- id (Primary Key)
- user_id (Foreign Key → User)
- image_path (file path, 500 chars)
- disease_name (200 chars)
- description (text)
- prevent (text - prevention tips)
- image_url (reference image URL)
- supplement_name (recommended product)
- supplement_image (product image URL)
- supplement_buy_link (e-commerce link)
- severity_level (mild/moderate/severe)
- confidence_score (float, 0-1)
- timestamp (DateTime, auto-set)
- Relationship: treatment_progress (1-to-many)
```

#### TreatmentProgress Model
```
- id (Primary Key)
- prediction_id (Foreign Key → Prediction)
- progress_date (DateTime)
- notes (text)
- improvement_status (worse/same/improving/recovered)
- follow_up_image (optional follow-up image path)
```

#### SavedSupplement Model
```
- id (Primary Key)
- user_id (Foreign Key → User)
- supplement_name (200 chars)
- supplement_image (URL)
- buy_link (e-commerce URL)
- disease_name (200 chars)
- saved_at (DateTime)
- Unique constraint: (user_id, supplement_name)
```

#### ForumTopic Model
```
- id (Primary Key)
- user_id (Foreign Key → User)
- title (200 chars)
- body (text - question content)
- created_at (DateTime)
- Relationship: replies (1-to-many)
```

#### ForumReply Model
```
- id (Primary Key)
- topic_id (Foreign Key → ForumTopic)
- user_id (Foreign Key → User)
- body (text - reply content)
- created_at (DateTime)
```

#### RemedySuggestion Model
```
- id (Primary Key)
- user_id (Foreign Key → User)
- title (200 chars)
- description (text)
- submitted_at (DateTime)
```

---

## 5. API ROUTES & ENDPOINTS

### Authentication Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/login` | User login with email and password |
| GET/POST | `/signup` | New user registration |
| GET | `/logout` | Clear session and logout |

### Core Features
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Homepage with feature cards |
| GET/POST | `/submit` | Upload image and get disease prediction |
| GET | `/index` | AI Engine landing page |
| GET | `/market` | Supplement marketplace |
| GET | `/dashboard` | User's prediction history |
| GET | `/saved-remedies` | Bookmarked supplement list |
| GET | `/prediction/<id>` | View single prediction details |
| GET/POST | `/add-progress/<id>` | Track treatment progress |

### Community Features
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/alerts` | Regional disease alerts |
| GET | `/forum` | Forum topic list |
| GET/POST | `/forum/new` | Create new forum topic |
| GET/POST | `/forum/topic/<id>` | View topic and post replies |
| GET | `/experts` | Directory of expert users |
| GET/POST | `/submit-remedy` | Share community remedies |
| GET | `/schemes` | Government subsidy information |

### Utility Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/contact` | Contact page |
| POST | `/bookmark-supplement/<name>` | Save supplement to favorites |
| POST | `/remove-bookmark/<id>` | Delete saved supplement |

---

## 6. MACHINE LEARNING PIPELINE

### Model Details
- **Architecture:** ResNet50 (pre-trained on ImageNet, finetuned for plant diseases)
- **Classes:** 39 disease categories across multiple crops
- **Input:** RGB images resized to 224×224 pixels
- **Preprocessing:** 
  - Image normalization using torchvision transforms
  - Tensor conversion
  - Unsqueeze (batch dimension addition)

### Inference Function
```python
def predict(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        pred = torch.argmax(output, dim=1).item()
    
    # Classify severity
    severity = get_severity_level(pred, confidence)
    
    return {
        'prediction': pred,
        'confidence': confidence,
        'severity': severity
    }
```

### Severity Classification
- **Mild:** Low confidence (<60%) → early-stage or uncertain detection
- **Moderate:** Default classification for medium confidence
- **Severe:** High confidence (>85%) + high-risk disease

---

## 7. USER INTERFACE IMPROVEMENTS

### Navigation & Layout
- **Responsive Navbar:** Bootstrap 5 with dynamic active state highlighting
- **Dropdown Menu:** Resources section removed; access through home page cards
- **Mobile Support:** Bootstrap grid system for responsive design
- **Active State:** Routes determine which nav item highlights (Home, Supplements, AI Engine)

### Form Design
- **Authentication Forms:** Centered card layout with clean input styling
- **Label Styling:** Removed dark background; now transparent with proper font weights
- **Upload Button:** Custom styled "Choose File" button with hover effects
- **Form Validation:** Client-side validation with Bootstrap error messaging

### Dashboard & History
- **Prediction Cards:** Grid layout showing past disease detections
- **Progress Timeline:** Treatment progress with status indicators (Improving/Recovered)
- **Action Buttons:** Consistent styling for view, edit, delete operations

### Community Pages
- **Feature Cards:** Five equal-width cards showcasing forums, experts, alerts
- **Topic Lists:** Display user, post date, reply count at a glance
- **Reply Threading:** Nested comment structure with clear author attribution

---

## 8. KEY FEATURES EXPLANATION

### 8.1 Disease Detection Workflow
1. User logs in and navigates to "AI Engine"
2. Selects plant leaf image
3. App processes image through ResNet50 model
4. Returns disease name, confidence score, and recommendations
5. Stores record in database with timestamp
6. User can save supplement or track progress

### 8.2 Treatment Tracking
1. User selects a past prediction from dashboard
2. Can add progress updates with notes and photos
3. Tracks status changes (Worse → Improving → Recovered)
4. Maintains complete history of recovery journey

### 8.3 Forum System
1. Browse existing discussion topics
2. Create new question with title and detailed body
3. Other users can post replies
4. Track reply count and discussion activity
5. Sort by creation date (newest first)

### 8.4 Expert Q&A
1. View directory of agricultural specialists
2. Contact experts via email
3. Get advice from professionals with specific expertise areas

### 8.5 Regional Alerts
1. View location-specific disease warnings
2. See recommended prevention measures
3. Stay informed about outbreak trends in farming region

### 8.6 Remedy Sharing
1. Users submit their own tested remedies
2. Other farmers can view and learn from experiences
3. Build community knowledge base over time

---

## 9. DATA FLOW EXAMPLES

### Example 1: Disease Detection & Bookmark
```
1. User uploads tomato_leaf.jpg
2. Route /submit processes image
3. ResNet50 predicts: "Early Blight" (92% confidence, Severe)
4. Database stores: Prediction + user_id + timestamp
5. Disease lookup from disease_info.csv gets description & supplements
6. User can click "Bookmark Supplement" → SavedSupplement record
7. Supplement appears in "My Remedies"
```

### Example 2: Forum Discussion
```
1. User posts: "Leaf yellowing in tomato plants"
2. ForumTopic created with user_id + title + body
3. User B replies with solution
4. ForumReply created linked to topic_id
5. Forum listing shows topic + reply count
6. Reply timestamp and author tracked
```

### Example 3: Treatment Progress Tracking
```
1. User opens prediction detail page
2. Associated TreatmentProgress records shown in timeline
3. User adds new update: "Leaves improving after fungicide"
4. Status: "Improving", note stored, optional follow-up photo
5. Timeline refreshes showing chronological progression
```

---

## 10. SECURITY CONSIDERATIONS

### Authentication
- Passwords hashed with Werkzeug `generate_password_hash()`
- Verification via `check_password_hash()`
- Session management with Flask-Login middleware
- Protected routes with `@login_required` decorator

### Data Validation
- WTForms for input sanitization
- Email format validation
- String length constraints (min/max validators)
- File upload path validation (saved to uploads folder)

### CSRF Protection
- Flask-WTF provides CSRF tokens on forms
- Hidden tag included in all POST forms

---

## 11. DATA SOURCES

### CSV Files
- **disease_info.csv:** 39 rows with disease names, descriptions, prevention tips, images
- **supplement_info.csv:** Matching supplement recommendations with product images and buy links

### Data Mapping
```
disease_index ↔ disease_info.csv row ↔ supplement_info.csv row
```

---

## 12. FILE STRUCTURE

```
App/
├── app.py                          # Main Flask application
├── CNN.py                          # CNN model definition (not actively used)
├── disease_info.csv                # Disease database
├── supplement_info.csv             # Product recommendations
├── requirements.txt                # Python dependencies
├── trained_model.pth               # Pre-trained weights (optional)
├── users.db                        # SQLite database (created at runtime)
├── static/
│   ├── uploads/                    # User-uploaded images
│   ├── uploads/progress/           # Treatment follow-up images
│   └── uploads/logo2.jpg           # App logo
└── templates/
    ├── base.html                   # Base layout with navbar
    ├── home.html                   # Homepage
    ├── login.html                  # Login form
    ├── signup.html                 # Registration form
    ├── index.html                  # AI Engine page
    ├── submit.html                 # Disease results display
    ├── dashboard.html              # User prediction history
    ├── saved_remedies.html         # Bookmarked supplements
    ├── prediction_detail.html       # Single prediction details
    ├── add_progress.html           # Treatment progress form
    ├── market.html                 # Supplement marketplace
    ├── contact-us.html             # Contact page
    ├── alerts.html                 # Regional disease alerts
    ├── forum.html                  # Forum topic list
    ├── new_forum_topic.html        # Create forum topic
    ├── forum_topic.html            # View topic & replies
    ├── experts.html                # Expert directory
    ├── submit_remedy.html          # Community remedy sharing
    └── schemes.html                # Government schemes
```

---

## 13. HOW TO RUN THE APPLICATION

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup Steps
```bash
# 1. Navigate to App directory
cd App

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python app.py
```

### Access Application
- Open browser and navigate to `http://127.0.0.1:5000`
- Create account or login
- Upload plant leaf image to test disease detection

---

## 14. KEY ACHIEVEMENTS

1. **Full Authentication System** - User registration, login, session management
2. **AI Model Integration** - ResNet50 disease detection with confidence scoring
3. **Prediction History** - Store and track all user predictions
4. **Treatment Tracking** - Record progress with status and optional photos
5. **Community Forum** - Threaded discussions with user-generated content
6. **Expert Directory** - Connect farmers with agricultural specialists
7. **Regional Alerts** - Location-specific disease warnings
8. **Remedy Sharing** - Community-contributed treatment knowledge
9. **Responsive UI** - Bootstrap 5 design with proper alignment and navigation
10. **Dynamic Active States** - Navigation highlights current page
11. **Database Persistence** - SQLAlchemy ORM with proper relationships
12. **Supplement Market** - Browse and bookmark recommended products

---

## 15. FUTURE ENHANCEMENTS

- [ ] Real-time disease alert notifications
- [ ] Expert response system for user questions
- [ ] Image comparison (before/after treatment)
- [ ] Mobile app version
- [ ] Advanced filtering in forum (by crop, region, disease)
- [ ] User reputation/rating system
- [ ] Weather-based alert system
- [ ] Export prediction reports as PDF
- [ ] Integration with government subsidy databases
- [ ] Multi-language support

---

## 16. CONCLUSION

CultiKure represents a comprehensive agricultural technology solution combining AI inference, community knowledge sharing, and expert connectivity. The system is designed to be user-friendly, scalable, and accessible to farmers of all technical backgrounds, enabling early disease detection and effective treatment planning through a unified platform.

**Key Innovation:** Bridging the gap between advanced AI technology and practical agricultural knowledge sharing through an integrated web platform.
