import streamlit as st
from PIL import Image
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Page configuration with updated parameter
st.set_page_config(
    page_title="NJ Transit Support",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.njtransit.com/',
        'Report a bug': None,
        'About': 'NJ Transit Support Assistant'
    }
)

# Cache model initialization
@st.cache_resource
def initialize_model():
    api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        raise ValueError("API key not found in Streamlit secrets")
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.7,  # Lower temperature for faster responses
        "top_p": 0.9,
        "top_k": 30,
        "max_output_tokens": 250,  # Reduced token limit
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

# Initialize model once
model = initialize_model()

# Custom CSS for responsive design
st.markdown("""
    <style>
    .responsive-title {
        font-size: clamp(1.5rem, 1.5vw + 1rem, 3rem);
        font-weight: bold;
        line-height: 1.2;
        padding-top: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    .stButton button {
        background-color: #dc3545;
        color: white;
        border: none;
        padding: 0.375rem 0.75rem;
        border-radius: 0.25rem;
        transition: background-color 0.15s ease-in-out;
    }
    
    .stButton button:hover {
        background-color: #c82333;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load and display the logo with title beside it
logo_path = "assets/new_jesry_transit_logo.png"
logo = Image.open(logo_path)

col1, col2 = st.columns([1, 3])

with col1:
    st.image(logo, use_container_width=True)

with col2:
    st.markdown('<p class="responsive-title">NJ Transit Support Assistant 💬</p>', unsafe_allow_html=True)

# Add a divider
st.markdown("<hr>", unsafe_allow_html=True)

# Load FAQs data
def load_faqs():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'FAQs_-_01042022.json')
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        faqs = []
        for section in data['iOSfaqs']['sections']:
            section_name = section['sec_name']
            for qa in section['sec_data']:
                faqs.append({
                    'section': section_name,
                    'question': qa['q'],
                    'answer': qa['a']
                })
        return faqs
    except FileNotFoundError:
        st.error("FAQ file not found. Please check the file path.")
        return []
    except json.JSONDecodeError:
        st.error("Error reading FAQ file. Please check the file format.")
        return []

def create_context_from_faqs(faqs):
    context = "You are an NJ Transit support assistant. Here are the official FAQs you should base your answers on:\n\n"
    for faq in faqs:
        context += f"Section: {faq['section']}\n"
        context += f"Q: {faq['question']}\n"
        context += f"A: {faq['answer']}\n\n"
    context += "\nPlease use this information to answer questions. If a question isn't covered in the FAQs, you can provide general help but mention that the information is not from the official FAQs."
    return context

# Additional transit data
TRANSIT_INFO = """
NJ TRANSIT is New Jersey's public transportation corporation. 
Key Services:
- Rail: 12 lines serving 165 stations
- Bus: Over 250 routes
- Light Rail: 3 systems (Hudson-Bergen, Newark, River LINE)

Common Rules:
- No eating/drinking on vehicles
- Quiet cars available on trains
- Off-peak fares available
- Monthly passes available
- Mobile ticketing through NJ TRANSIT app
- Service animals allowed
- Bicycles permitted with restrictions
"""

def check_faq_relevance(prompt, faqs):
    """Check if the prompt is related to any FAQ"""
    for faq in faqs:
        if any(word.lower() in faq['question'].lower() for word in prompt.split()):
            return True
    return False

def get_chatbot_response(prompt, conversation_history, faqs_context):
    """Get response from Gemini model with FAQs or general knowledge"""
    try:
        is_faq_related = check_faq_relevance(prompt, st.session_state.faqs)
        
        if is_faq_related:
            # Use FAQ context for NJ Transit specific questions
            chat = model.start_chat(history=[])
            chat.send_message(faqs_context)
        else:
            # Use Gemini's knowledge for non-FAQ questions
            chat = model.start_chat(history=[])
            chat.send_message("""You are a helpful NJ Transit assistant. For questions not covered in the FAQs, 
            provide accurate information based on your knowledge about NJ Transit's current policies and services. 
            Be specific and helpful while maintaining accuracy.""")
        
        # Add recent conversation history
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        for msg in recent_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                chat.send_message(msg["content"])
        
        response = chat.send_message(prompt)
        return response.text

    except Exception as e:
        return f"I apologize, but I'm having trouble responding right now. Please try again later. Error: {str(e)}"

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "faqs" not in st.session_state:
    st.session_state.faqs = load_faqs()

if "faqs_context" not in st.session_state:
    if st.session_state.faqs:
        st.session_state.faqs_context = create_context_from_faqs(st.session_state.faqs)
    else:
        st.session_state.faqs_context = "You are an NJ Transit support assistant. Please provide general help."

# Example questions
st.markdown("""
<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='color: #0066cc; margin-bottom: 10px;'>Example questions you can ask:</h4>
    <ul style='margin-bottom: 0;'>
        <li>How do I purchase tickets using the NJ TRANSIT Mobile App?</li>
        <li>What payment methods are accepted in the app?</li>
        <li>How do I activate my ticket?</li>
        <li>What should I do if my ticket doesn't scan properly?</li>
        <li>How do I create a My Transit account?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Add a divider
st.markdown("<hr>", unsafe_allow_html=True)

# Main chat container
container = st.container()
with container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f'<div class="timestamp">{message["timestamp"]}</div>', 
                          unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("How can I help you with NJ Transit today?"):
        current_time = datetime.now().strftime("%I:%M %p")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": current_time
        })
        
        # Get chatbot response
        response = get_chatbot_response(
            prompt, 
            st.session_state.messages, 
            st.session_state.faqs_context
        )
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": current_time
        })
        
        st.rerun()

# Clear chat button with better positioning
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🗑️ Clear Chat", help="Clear chat history"):
        st.session_state.messages = []
        st.rerun()