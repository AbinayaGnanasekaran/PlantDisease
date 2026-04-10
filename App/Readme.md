# Plant Disease Detection System

This is an AI-powered plant disease detection system built with Flask.

## Features

- **User Authentication**: Secure login and signup system
- **Disease Detection**: AI-powered analysis of plant leaf images
- **Supplement Recommendations**: Get supplement suggestions based on detected diseases
- **Marketplace**: Browse available supplements

## Authentication System

The application now includes a complete user authentication system:

### User Registration
- Visit `/signup` to create a new account
- Required fields: Username, Email, Password
- Password must be at least 6 characters long

### User Login
- Visit `/login` to sign in
- Use your email and password to authenticate
- Option to "Remember Me" for persistent sessions

### Protected Routes
- `/index` (AI Engine) - Requires login
- `/submit` (Disease Prediction) - Requires login
- `/logout` - Logs out current user

### Navigation
- Unauthenticated users see Login/Signup links
- Authenticated users see Welcome message and Logout link
- Home page shows appropriate call-to-action based on auth status

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to `http://127.0.0.1:5000`

## Database

The application uses SQLite database (`users.db`) to store user accounts. The database is automatically created when you first run the application.

## Security Notes

- Change the `SECRET_KEY` in production
- Use HTTPS in production
- Consider using environment variables for sensitive configuration
 
