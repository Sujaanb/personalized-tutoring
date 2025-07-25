import streamlit as st
import json
import os
from typing import Dict, Any
import tempfile
from pathlib import Path
import time
import plotly.graph_objects as go
import plotly.express as px
import hashlib
import datetime
from dataclasses import dataclass

# Import your existing modules
from main import RAGAssistant
from config import config

@dataclass
class User:
    username: str
    email: str
    password_hash: str
    created_at: str
    study_streak: int = 0
    total_questions: int = 0

class StreamlitRAGInterface:
    """Streamlit interface for the RAG Assistant."""
    
    def __init__(self):
        self.assistant = None
        
    def initialize_assistant(self):
        """Initialize the RAG assistant if not already done."""
        if self.assistant is None:
            try:
                self.assistant = RAGAssistant()
                return True
            except Exception as e:
                st.error(f"Failed to initialize assistant: {str(e)}")
                return False
        return True

class AuthSystem:
    """Handle user authentication and management."""
    
    def __init__(self):
        self.users_file = "users.json"
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        except:
            self.users = {}
    
    def save_users(self):
        """Save users to JSON file."""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, email: str, password: str, user_prefs: dict = None) -> bool:
        """Register a new user."""
        if username in self.users:
            return False
        
        user_data = {
            "username": username,
            "email": email,
            "password_hash": self.hash_password(password),
            "created_at": datetime.datetime.now().isoformat(),
            "study_streak": 0,
            "total_questions": 0
        }
        
        # Add user preferences if provided
        if user_prefs:
            user_data.update(user_prefs)
        
        self.users[username] = user_data
        self.save_users()
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user login."""
        if username in self.users:
            return self.users[username]["password_hash"] == self.hash_password(password)
        return False
    
    def get_user(self, username: str) -> dict:
        """Get user data."""
        return self.users.get(username, {})

def show_signup_page():
    """Display the beautiful sign-up page."""
    st.markdown("""
    <style>
    .signup-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 2rem 0;
        text-align: center;
    }
    
    .signup-header {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-weight: bold;
    }
    
    .signup-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .signup-form {
        background: rgba(255,255,255,0.95);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        margin: 2rem auto;
        max-width: 400px;
    }
    
    .form-title {
        color: #667eea;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    
    .benefits-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 0.5rem;
        background: rgba(255,255,255,0.7);
        border-radius: 10px;
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        width: 40px;
        text-align: center;
    }
    
    .login-link {
        text-align: center;
        margin-top: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
    }
    
    .testimonial {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main signup container
    st.markdown("""
    <div class="signup-container">
        <h1 class="signup-header">🎓 Welcome to SAGE</h1>
        <p class="signup-subtitle">Your Personal AI Learning Assistant</p>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">Join thousands of learners who are already mastering new subjects with AI-powered guidance!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sign-up form
        st.markdown("""
        <div class="signup-form">
            <h2 class="form-title">Create Your Account</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("signup_form", clear_on_submit=False):
            st.markdown("### 📝 Personal Information")
            
            username = st.text_input(
                "👤 Username",
                placeholder="Choose a unique username",
                help="This will be your unique identifier"
            )
            
            email = st.text_input(
                "📧 Email Address",
                placeholder="your.email@example.com",
                help="We'll use this for important updates"
            )
            
            st.markdown("### 🔒 Security")
            
            password = st.text_input(
                "🔑 Password",
                type="password",
                placeholder="Create a strong password",
                help="Use at least 8 characters with numbers and symbols"
            )
            
            confirm_password = st.text_input(
                "🔑 Confirm Password",
                type="password",
                placeholder="Confirm your password"
            )
            
            st.markdown("### 🎯 Learning Goals")
            
            learning_goal = st.selectbox(
                "🎓 What's your main learning goal?",
                [
                    "Academic Studies",
                    "Professional Development",
                    "Personal Interest",
                    "Exam Preparation",
                    "Research Projects",
                    "Skill Enhancement"
                ]
            )
            
            # Student class selection (optional)
            st.markdown("### 📚 Academic Information (Optional)")
            
            student_class = st.selectbox(
                "🏫 Student Class (if applicable)",
                [
                    "Not a student / Not applicable",
                    "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6",
                    "Class 7", "Class 8", "Class 9", "Class 10", "Class 11", "Class 12",
                    "Undergraduate", "Graduate", "Postgraduate"
                ],
                help="Select your current academic level if you're a student"
            )
            
            subjects = st.multiselect(
                "� What subjects interest you?",
                [
                    "Mathematics", "Science", "History", "Literature",
                    "Programming", "Business", "Medicine", "Law",
                    "Engineering", "Arts", "Languages", "Psychology",
                    "Physics", "Chemistry", "Biology", "Geography",
                    "Economics", "Philosophy", "Music", "Sports"
                ]
            )
            
            # Terms and conditions
            st.markdown("---")
            terms_accepted = st.checkbox(
                "✅ I agree to the Terms of Service and Privacy Policy",
                help="By checking this, you agree to our terms and conditions"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Create My Account",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                # Validation
                if not all([username, email, password, confirm_password]):
                    st.error("❌ Please fill in all required fields!")
                elif password != confirm_password:
                    st.error("❌ Passwords don't match!")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters long!")
                elif not terms_accepted:
                    st.error("❌ Please accept the terms and conditions!")
                elif '@' not in email:
                    st.error("❌ Please enter a valid email address!")
                else:
                    # Prepare user preferences
                    user_prefs = {
                        "learning_goal": learning_goal,
                        "student_class": student_class,
                        "subjects": subjects,
                        "signup_date": datetime.datetime.now().isoformat()
                    }
                    
                    # Try to register user
                    auth = AuthSystem()
                    if auth.register_user(username, email, password, user_prefs):
                        st.success("🎉 Account created successfully!")
                        st.balloons()
                        
                        # Store in session state
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_prefs = user_prefs
                        st.session_state.student_class = student_class
                        
                        st.info("🔄 Redirecting to your dashboard...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Username already exists! Please choose a different one.")
    
    with col2:
        # Benefits and features with illustrations
        st.markdown("""
        <div class="benefits-card">
            <h3 style="color: #d63031; text-align: center; margin-bottom: 1.5rem;">🌟 Why Choose SAGE?</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Educational level illustration
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">🎒 Perfect for All Learning Levels</h4>
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                <div style="margin: 0.5rem;">
                    <div style="font-size: 2rem;">🧒</div>
                    <small style="color: #34495e; font-weight: bold;">Primary School</small>
                </div>
                <div style="margin: 0.5rem;">
                    <div style="font-size: 2rem;">👨‍🎓</div>
                    <small style="color: #34495e; font-weight: bold;">High School</small>
                </div>
                <div style="margin: 0.5rem;">
                    <div style="font-size: 2rem;">🎓</div>
                    <small style="color: #34495e; font-weight: bold;">College</small>
                </div>
                <div style="margin: 0.5rem;">
                    <div style="font-size: 2rem;">👩‍💼</div>
                    <small style="color: #34495e; font-weight: bold;">Professional</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning journey illustration
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
            <h4 style="color: #2d3436; margin-bottom: 1rem;">🚀 Your Learning Journey</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📚</div>
                    <small style="color: #2d3436; font-weight: bold;">Upload Materials</small>
                </div>
                <div style="font-size: 1.5rem; color: #6c5ce7;">➡️</div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                    <small style="color: #2d3436; font-weight: bold;">AI Analysis</small>
                </div>
                <div style="font-size: 1.5rem; color: #6c5ce7;">➡️</div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💡</div>
                    <small style="color: #2d3436; font-weight: bold;">Smart Learning</small>
                </div>
                <div style="font-size: 1.5rem; color: #6c5ce7;">➡️</div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏆</div>
                    <small style="color: #2d3436; font-weight: bold;">Master Topics</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        features = [
            ("🧠", "AI-Powered Learning", "Get personalized explanations and insights"),
            ("📚", "Smart Document Processing", "Upload and analyze any study material"),
            ("🎯", "Interactive Quizzes", "Test your knowledge with auto-generated questions"),
            ("📊", "Progress Tracking", "Monitor your learning journey and achievements"),
            ("💬", "24/7 AI Tutor", "Ask questions anytime, get instant answers"),
            ("🏆", "Gamified Learning", "Earn streaks and celebrate milestones")
        ]
        
        for icon, title, description in features:
            st.markdown(f"""
            <div class="feature-item">
                <div class="feature-icon">{icon}</div>
                <div>
                    <strong style="color: #667eea;">{title}</strong><br>
                    <small style="color: #666;">{description}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Testimonial
        st.markdown("""
        <div class="testimonial">
            <h4>💬 What Our Users Say</h4>
            <p>"SAGE transformed how I study! The AI explanations are so clear, and the quiz feature helps me retain everything better."</p>
            <strong>- Sarah, Graduate Student</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
            <h4 style="color: #2e7d32; margin-bottom: 1rem;">📈 Join Our Community</h4>
            <div style="display: flex; justify-content: space-around;">
                <div>
                    <h3 style="color: #1976d2; margin: 0;">10K+</h3>
                    <small>Active Learners</small>
                </div>
                <div>
                    <h3 style="color: #1976d2; margin: 0;">50K+</h3>
                    <small>Questions Answered</small>
                </div>
                <div>
                    <h3 style="color: #1976d2; margin: 0;">95%</h3>
                    <small>Success Rate</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Login link
    st.markdown("""
    <div class="login-link">
        <p style="margin: 0; color: #667eea;">Already have an account? 
        <strong style="color: #1976d2;">Click the 'Login' option in the sidebar!</strong></p>
    </div>
    """, unsafe_allow_html=True)

def show_login_page():
    """Display the login page."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0; color: white;">
        <h1>🔐 Welcome Back to SAGE</h1>
        <p>Sign in to continue your learning journey</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h2 style="color: #667eea; text-align: center; margin-bottom: 1.5rem;">Sign In</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            
            if submitted:
                auth = AuthSystem()
                if auth.authenticate_user(username, password):
                    st.success("✅ Login successful!")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    
                    # Load user data
                    user_data = auth.get_user(username)
                    st.session_state.study_streak = user_data.get("study_streak", 0)
                    st.session_state.total_questions_answered = user_data.get("total_questions", 0)
                    st.session_state.student_class = user_data.get("student_class", "Not specified")
                    st.session_state.learning_goal = user_data.get("learning_goal", "General Learning")
                    st.session_state.subjects = user_data.get("subjects", [])
                    
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password!")
        
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 10px;">
            <p>Don't have an account? Select 'Sign Up' from the sidebar!</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="SAGE - Your Personal Learning Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = "signup"
    
    # Authentication sidebar
    if not st.session_state.logged_in:
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <h3 style="color: white; margin: 0;">🎓 SAGE</h3>
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">AI Learning Assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            auth_option = st.radio(
                "Choose an option:",
                ["Sign Up", "Login"],
                horizontal=True
            )
            
            if auth_option == "Sign Up":
                st.session_state.auth_page = "signup"
            else:
                st.session_state.auth_page = "login"
            
            # Quick info
            st.markdown("""
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">✨ Get Started</h4>
                <ul style="margin: 0; padding-left: 1rem; color: #666;">
                    <li>Create your free account</li>
                    <li>Upload study materials</li>
                    <li>Start learning with AI</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Show appropriate page
        if st.session_state.auth_page == "signup":
            show_signup_page()
        else:
            show_login_page()
        
        return  # Exit here if not logged in
    
    # Rest of the main app (existing code)
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: #1e3a8a;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #1e40af;
        font-size: 1.2rem;
        margin: 0;
    }
    
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .quiz-container {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #1e40af;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        padding: 1rem;
        border-radius: 10px;
        color: #1e40af;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: #1e40af;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .quiz-question {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        height: 10px;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize the interface
    if 'interface' not in st.session_state:
        st.session_state.interface = StreamlitRAGInterface()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize quiz state
    if 'quiz_active' not in st.session_state:
        st.session_state.quiz_active = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = {'correct': 0, 'total': 0}
    if 'study_streak' not in st.session_state:
        st.session_state.study_streak = 0
    if 'total_questions_answered' not in st.session_state:
        st.session_state.total_questions_answered = 0
    
    # Main header with attractive design and user welcome
    username = st.session_state.get('username', 'User')
    student_class = st.session_state.get('student_class', 'Not specified')
    
    # Create personalized welcome message
    if student_class != "Not a student / Not applicable" and student_class != "Not specified":
        welcome_subtitle = f"Your Personal Learning Assistant - Tailored for {student_class} Level"
    else:
        welcome_subtitle = "Your Personal Learning Assistant - Master Any Subject with AI-Powered Guidance"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🎓 Welcome back, {username}!</h1>
        <p>{welcome_subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        # User profile section with class information
        student_class = st.session_state.get('student_class', 'Not specified')
        learning_goal = st.session_state.get('learning_goal', 'General Learning')
        
        st.markdown(f"""
        <div class="sidebar-section">
            <h3 style="text-align: center; color: #667eea; margin-bottom: 1rem;">👋 Hello, {username}!</h3>
            <p style="text-align: center; color: #666; font-size: 0.9rem;">Ready to learn something new today?</p>
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 0.8rem; border-radius: 8px; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.85rem; color: #1976d2;">
                    <strong>🎓 Level:</strong> {student_class}<br>
                    <strong>🎯 Goal:</strong> {learning_goal}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            # Save user data before logout
            auth = AuthSystem()
            if username in auth.users:
                auth.users[username]["study_streak"] = st.session_state.get("study_streak", 0)
                auth.users[username]["total_questions"] = st.session_state.get("total_questions_answered", 0)
                # Update user preferences if they exist
                if hasattr(st.session_state, 'student_class'):
                    auth.users[username]["student_class"] = st.session_state.get("student_class", "Not specified")
                if hasattr(st.session_state, 'learning_goal'):
                    auth.users[username]["learning_goal"] = st.session_state.get("learning_goal", "General Learning")
                if hasattr(st.session_state, 'subjects'):
                    auth.users[username]["subjects"] = st.session_state.get("subjects", [])
                auth.save_users()
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        # User stats at the top
        st.markdown("""
        <div class="sidebar-section">
            <h2 style="text-align: center; color: #667eea; margin-bottom: 1rem;">📊 Your Progress</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Study statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 Study Streak", f"{st.session_state.study_streak} days")
        with col2:
            st.metric("✅ Questions", st.session_state.total_questions_answered)
        
        # Progress visualization
        if st.session_state.total_questions_answered > 0:
            recent_performance = min(100, (st.session_state.quiz_score.get('correct', 0) / max(1, st.session_state.quiz_score.get('total', 1))) * 100)
            st.progress(recent_performance / 100)
            st.caption(f"Recent Quiz Performance: {recent_performance:.1f}%")
        
        st.markdown("---")
        
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="color: #667eea; text-align: center;">📚 Knowledge Base</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize assistant
        if not st.session_state.interface.initialize_assistant():
            st.stop()
        
        assistant = st.session_state.interface.assistant
        
        # File upload section with better styling
        with st.expander("📁 Upload New Documents", expanded=False):
            st.markdown("**Drag and drop your study materials here:**")
            uploaded_files = st.file_uploader(
                "Choose PDF or TXT files",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="📖 Upload textbooks, notes, or any study materials",
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                st.info(f"📄 {len(uploaded_files)} file(s) ready to upload")
                if st.button("🚀 Process Files", use_container_width=True):
                    with st.spinner("🔄 Processing your documents..."):
                        success_count = process_uploaded_files(uploaded_files, assistant)
                        if success_count > 0:
                            st.balloons()  # Celebrate successful upload
                            st.success(f"🎉 Successfully processed {success_count} file(s)!")
        
        # Quick upload buttons with emojis
        st.markdown("**Quick Actions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Process PDFs", use_container_width=True):
                with st.spinner("Processing PDF files..."):
                    result = assistant.upload_service.upload_pdf_files(config.uploads_directory)
                    if result['success']:
                        st.success("✅ " + result['message'])
                    else:
                        st.error("❌ " + result['message'])
        
        with col2:
            if st.button("📝 Process TXTs", use_container_width=True):
                with st.spinner("Processing TXT files..."):
                    result = assistant.upload_service.upload_txt_files(config.uploads_directory)
                    if result['success']:
                        st.success("✅ " + result['message'])
                    else:
                        st.error("❌ " + result['message'])
        
        # Knowledge base status with visual elements
        with st.expander("📊 Knowledge Base Status", expanded=False):
            if st.button("🔄 Refresh Status", use_container_width=True):
                status = assistant.vector_manager.get_knowledge_base_status()
                if status['success']:
                    # Create a simple visualization
                    if status['total_chunks'] > 0:
                        st.metric("📚 Total Knowledge Chunks", status['total_chunks'])
                        
                        if status['file_types']:
                            # Create a pie chart for file types
                            file_types = status['file_types']
                            if len(file_types) > 1:
                                fig = px.pie(
                                    values=list(file_types.values()),
                                    names=[f"{k.upper()} Files" for k in file_types.keys()],
                                    title="Document Types in Knowledge Base"
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                for file_type, count in file_types.items():
                                    emoji = "📄" if file_type == 'pdf' else "📝"
                                    st.write(f"{emoji} {file_type.upper()}: {count} chunks")
                    else:
                        st.info("🗂️ Knowledge base is empty. Upload some documents to get started!")
                else:
                    st.error(f"❌ Error: {status.get('error', 'Unknown error')}")
        
        # List files section
        with st.expander("📂 Available Files", expanded=False):
            if st.button("📋 Show My Files", use_container_width=True):
                file_info = assistant.upload_service.list_available_files(config.uploads_directory)
                if file_info['exists']:
                    if file_info['pdf_files']:
                        st.markdown("**📄 PDF Files:**")
                        for filename in file_info['pdf_files']:
                            st.markdown(f"• {filename}")
                    if file_info['txt_files']:
                        st.markdown("**📝 Text Files:**")
                        for filename in file_info['txt_files']:
                            st.markdown(f"• {filename}")
                    if file_info['total_files'] == 0:
                        st.info("📭 No files found. Upload some documents to get started!")
                else:
                    st.error(file_info['message'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="chat-container">
            <h2 style="color: #667eea; text-align: center; margin-bottom: 1rem;">💬 Chat with Your AI Tutor</h2>
            <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">Ask questions about your study materials and get instant, intelligent answers!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history with better styling
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: #1e40af; margin: 1rem 0;">
                    <h3>👋 Welcome to Your AI Tutor!</h3>
                    <p>Upload your study materials and start asking questions. I'm here to help you learn better!</p>
                    <p><strong>Try asking:</strong></p>
                    <p>• "Explain this concept in simple terms"</p>
                    <p>• "Give me examples of this topic"</p>
                    <p>• "What are the key points I should remember?"</p>
                </div>
                """, unsafe_allow_html=True)
            
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                # User message with custom styling
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin: 0.5rem 0; color: white;">
                    <strong>🧑‍🎓 You:</strong> {user_msg}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response with custom styling
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 1rem; border-radius: 15px; margin: 0.5rem 0;">
                    <strong style="color: #2c3e50;">🤖 AI Tutor:</strong> <span style="color: #2c3e50;">{bot_msg}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced chat input
        st.markdown("---")
        if prompt := st.chat_input("💭 What would you like to learn today?", key="main_chat"):
            if not st.session_state.quiz_active:
                # Add user message to chat history
                with st.spinner("🤔 Let me think about that..."):
                    try:
                        # Get user context for personalized responses
                        student_class = st.session_state.get('student_class', 'Not specified')
                        learning_goal = st.session_state.get('learning_goal', 'General Learning')
                        
                        # Enhance the prompt with user context
                        if student_class != "Not a student / Not applicable" and student_class != "Not specified":
                            context_prompt = f"[Student Level: {student_class}, Learning Goal: {learning_goal}] {prompt}"
                        else:
                            context_prompt = f"[Learning Goal: {learning_goal}] {prompt}"
                        
                        response = assistant.process_user_input(context_prompt)
                        
                        # Add to chat history (show original prompt to user)
                        st.session_state.chat_history.append((prompt, response))
                        st.session_state.total_questions_answered += 1
                        
                        # Update study streak
                        if st.session_state.total_questions_answered % 5 == 0:
                            st.session_state.study_streak += 1
                            st.balloons()
                            st.success(f"🎉 Great job! Study streak: {st.session_state.study_streak} days!")
                        
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"😅 Oops! I encountered an issue: {str(e)}"
                        st.session_state.chat_history.append((prompt, error_msg))
                        st.rerun()
            else:
                st.warning("🎯 Please finish your current quiz before asking new questions!")
    
    with col2:
        st.markdown("""
        <div class="quiz-container">
            <h2 style="color: #d63031; text-align: center; margin-bottom: 1rem;">🎯 Quiz Challenge</h2>
            <p style="text-align: center; color: #333; margin-bottom: 1rem;">Test your knowledge and track your progress!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.quiz_active:
            # Quiz start section with motivation
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                <h4 style="color: #FF6B35; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🧠 Ready to Challenge Yourself?</h4>
                <p style="color: #000000; font-weight: 500;">Test your understanding with AI-generated questions based on your study materials!</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎲 Start Quiz Challenge", use_container_width=True, type="primary"):
                # Check if knowledge base has content
                status = assistant.vector_manager.get_knowledge_base_status()
                if status.get('total_chunks', 0) > 0:
                    st.session_state.quiz_active = True
                    st.session_state.quiz_score = {'correct': 0, 'total': 0}
                    # Generate first question
                    generate_quiz_question(assistant)
                    st.rerun()
                else:
                    st.error("📚 Knowledge base is empty. Please upload documents first!")
            
            # Display some motivational stats
            if st.session_state.total_questions_answered > 0:
                st.markdown("### 📈 Your Learning Journey")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = min(100, (st.session_state.total_questions_answered / 50) * 100),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Learning Progress"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightblue"},
                            {'range': [75, 100], 'color': "blue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            # Active quiz section with better styling
            if st.session_state.current_question:
                question_data = st.session_state.current_question
                
                st.markdown(f"""
                <div class="quiz-question">
                    <h4 style="color: #000000;">📝 Question {st.session_state.quiz_score['total'] + 1}</h4>
                    <p><strong style="color: #000000;">{question_data["question"]}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Answer options with better styling
                answer = st.radio(
                    "Choose your answer:",
                    question_data["options"],
                    key=f"quiz_answer_{st.session_state.quiz_score['total']}",
                    label_visibility="collapsed"
                )
                
                col_submit, col_end = st.columns(2)
                
                with col_submit:
                    if st.button("✅ Submit Answer", use_container_width=True, type="primary"):
                        # Check answer
                        selected_letter = answer[0] if answer else ""
                        correct_letter = question_data["correct_answer"].upper()
                        
                        st.session_state.quiz_score['total'] += 1
                        
                        if selected_letter == correct_letter:
                            st.session_state.quiz_score['correct'] += 1
                            st.success("🎉 Excellent! Correct answer!")
                            st.balloons()
                        else:
                            st.error(f"🤔 Not quite right. The correct answer was {correct_letter}.")
                            st.info("💡 Keep learning! Every mistake is a step towards mastery.")
                        
                        # Generate next question
                        time.sleep(1)  # Brief pause for feedback
                        generate_quiz_question(assistant)
                        st.rerun()
                
                with col_end:
                    if st.button("🛑 End Quiz", use_container_width=True):
                        end_quiz()
                        st.rerun()
                
                # Enhanced score display
                score = st.session_state.quiz_score
                if score['total'] > 0:
                    percentage = (score['correct'] / score['total']) * 100
                    
                    # Progress bar for current quiz
                    st.markdown("### 🏆 Current Quiz Score")
                    progress = score['correct'] / max(score['total'], 1)
                    st.progress(progress)
                    
                    # Score with color coding
                    color = "#27AE60" if percentage >= 80 else "#F39C12" if percentage >= 60 else "#E74C3C"
                    st.markdown(f"""
                    <div style="background: {color}; color: #1e40af; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                        <h3>{score['correct']}/{score['total']} ({percentage:.1f}%)</h3>
                        <p>{'🌟 Outstanding!' if percentage >= 90 else '👍 Great job!' if percentage >= 70 else '💪 Keep trying!'}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Action buttons section
        st.markdown("---")
        st.markdown("### 🛠️ Quick Actions")
        
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("✨ Chat cleared! Ready for a fresh start!")
            time.sleep(1)
            st.rerun()
        
        if st.button("🔄 Reset Progress", use_container_width=True):
            st.session_state.study_streak = 0
            st.session_state.total_questions_answered = 0
            st.session_state.quiz_score = {'correct': 0, 'total': 0}
            st.info("📊 Progress reset! Time for a new learning journey!")
            time.sleep(1)
            st.rerun()
    
    # Footer with credits and tips
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 2rem; color: #1e40af;">
        <h3>🎓 Happy Learning!</h3>
    </div>
    """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files, assistant):
    """Process uploaded files with better error handling and user feedback."""
    success_count = 0
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Move to uploads directory
            uploads_path = Path(config.uploads_directory)
            uploads_path.mkdir(exist_ok=True)
            final_path = uploads_path / uploaded_file.name
            
            # Copy file to uploads directory
            import shutil
            shutil.move(tmp_path, final_path)
            success_count += 1
            
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {str(e)}")
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    if success_count > 0:
        # Process the uploaded files
        result = assistant.upload_service.upload_documents(config.uploads_directory)
        if result['success']:
            return success_count
        else:
            st.error(f"❌ {result['message']}")
            return 0
    return 0

def generate_quiz_question(assistant):
    """Generate a new quiz question with better error handling."""
    try:
        contexts = assistant.vector_manager.get_random_knowledge_chunks(num_chunks=1)
        if contexts:
            # Get user context for appropriate difficulty level
            student_class = st.session_state.get('student_class', 'Not specified')
            
            # Add context about student level for appropriate question difficulty
            if student_class != "Not a student / Not applicable" and student_class != "Not specified":
                context_with_level = f"[Generate question appropriate for {student_class} level] {contexts[0]}"
                question_data = assistant._generate_quiz_question(context_with_level)
            else:
                question_data = assistant._generate_quiz_question(contexts[0])
                
            if question_data:
                st.session_state.current_question = question_data
            else:
                st.error("🤔 Could not generate a question. Please try again.")
        else:
            st.error("📚 Could not retrieve context for a question. Make sure you have uploaded some documents.")
    except Exception as e:
        st.error(f"⚠️ Error generating question: {str(e)}")

def end_quiz():
    """End the current quiz session with celebration."""
    score = st.session_state.quiz_score
    if score['total'] > 0:
        percentage = (score['correct'] / score['total']) * 100
        
        # Celebratory message based on performance
        if percentage >= 90:
            message = "🌟 Outstanding performance! You're a learning superstar!"
            st.balloons()
        elif percentage >= 80:
            message = "🎉 Excellent work! You really know your stuff!"
        elif percentage >= 70:
            message = "👍 Great job! You're making solid progress!"
        elif percentage >= 60:
            message = "💪 Good effort! Keep studying and you'll improve!"
        else:
            message = "📚 Don't worry! Every expert was once a beginner. Keep learning!"
        
        st.success(f"🏁 Quiz completed! Final score: {score['correct']}/{score['total']} ({percentage:.1f}%)")
        st.info(message)
    
    st.session_state.quiz_active = False
    st.session_state.current_question = None
    st.session_state.quiz_score = {'correct': 0, 'total': 0}

if __name__ == "__main__":
    main()