import streamlit as st
st.set_page_config(page_title="Smart Patient Medicine Tracker", layout="wide")
from twilio.rest import Client

import cv2
import face_recognition
import numpy as np
import pandas as pd
import qrcode
from PIL import Image
import os
import random
import time
import io
from datetime import datetime
import json
import base64
from pyzbar import pyzbar
import matplotlib.pyplot as plt
import seaborn as sns
import decode
from datetime import datetime

# Custom CSS for animations and styling
st.markdown("""
<style>
      .sidebar .sidebar-content {
    background-image: linear-gradient(#4CAF50, #81C784);
    color: white;
}

.big-font {
    font-size: 35px !important;
    font-weight: bold;
    animation: fadeIn 1.5s;
    color: #ffffff;
    text-align: center;
    margin-bottom: 20px;
}
.sidebar{
    background-color: #3e62e6;
}

.menu-item {
    
  font-size: 24px !important;
    font-weight: bold;
    padding: 15px 20px;
    margin: 10px 0;
    border-radius: 10px;
    transition: all 0.3s;
    background-color: rgba(195, 190, 190, 0.329);
    border: 2px solid #4CAF50;
    cursor: pointer;
    color: #4CAF50;
    text-align: center;
    text-decoration: none;
    display: block;
}

.menu-item:hover {
    background-color: #4CAF50;
    color: white;
    transform: scale(1.02);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.menu-item.active {
    background-color: #4CAF50;
    color: white;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.menu-item:hover {
    background-color: rgba(255,255,255,0.1);
    transform: scale(1.02);
}

.page-title {
    font-size: 40px !important;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
    margin: 20px 0;
    animation: slideInDown 1s;
}

.section-title {
    font-size: 28px !important;
    color: #4CAF50;
    margin: 15px 0;
    animation: fadeIn 1s;
}

.instruction-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    animation: slideIn 1s;
    color: #000000;
    box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
    transition: transform 0.3s;
}

.instruction-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.process-step {
    padding: 10px;
    margin: 5px 0;
    border-left: 3px solid #4CAF50;
    background-color: rgba(76, 175, 80, 0.1);
}

.stTable {
    background-color: white !important;
}

.stTable th {
    background-color: #4CAF50 !important;
    color:  #063a09 !important;
}

.stTable td {
    color:  #063a09 !important;
}

.button-primary {
    background-color: #4CAF50;
    color: white;
    padding: 12px 24px;
    border-radius: 5px;
    border: none;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.3s;
}

.button-primary:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}

.stButton>button {
    background-color: #4CAF50 !important;
    color: white !important;
    font-size: 18px !important;
    padding: 12px 24px !important;
    border-radius: 5px !important;
    transition: all 0.3s !important;
}

.stButton>button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
}

@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

@keyframes slideIn {
    0% { transform: translateX(-100%); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}

@keyframes slideInDown {
    0% { transform: translateY(-50px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)
# Add these Streamlit secrets in your secrets.toml file or use st.secrets
def send_purchase_sms(phone_number, patient_name, medicine_name, bill_number, total_amount, timestamp):
    """
    Send SMS confirmation to patient after medicine purchase
    """
    try:
        # Import Twilio
        from twilio.rest import Client
        
        # Try to get Twilio credentials from Streamlit secrets
        try:
            TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
            TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
            TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]
        except Exception as e:
            # If you have hardcoded values as fallback, use them here
            # For example:
            TWILIO_ACCOUNT_SID = "ACc4c71b1cdd6c42af1128bd4497facd40"
            TWILIO_AUTH_TOKEN = "2422692c773ba4cb32fe651271f4440a"
            TWILIO_PHONE_NUMBER = "+13252464246"

        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Format message
        message_body = (
                f"Dear {patient_name},\n\n"
                f"Your medicine purchase was successful!\n"
                f"Medicine: {medicine_name}\n"
                
                f"Amount: ₹{total_amount}\n"
                f"Time: {timestamp}\n\n"
                f"Thank you for purchasing -spmt (kawin)"
            )


        # Send message
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=f"+91{phone_number}"  # Assuming Indian phone numbers
        )

        return True, "SMS sent successfully!"
    except ImportError:
        return False, "Twilio package is not installed"
    except Exception as e:
        return False, f"Error sending SMS: {str(e)}"

def show_process_step(title, description):
    st.markdown(f"""
    <div class="process-step">
        <strong>{title}</strong><br>
        {description}
    </div>
    """, unsafe_allow_html=True)

def generate_qr_code(data, filename):
    try:
        qr = qrcode.QRCode(version=1, box_size=5, border=5)
        qr.add_data(data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(filename)
        return True
    except Exception as e:
        st.error(f"Error generating QR code: {str(e)}")
        return False

def show_face_encoding_process(img_array):
    with st.spinner("Processing face..."):
        col1, col2 = st.columns(2)
        with col1:
            show_process_step("Step 1: Image Loading", "Converting image to numerical array")
            time.sleep(0.5)
            
            show_process_step("Step 2: Face Detection", "Locating face in the image")
            face_locations = face_recognition.face_locations(img_array)
            time.sleep(0.5)
            
            show_process_step("Step 3: Feature Extraction", "Identifying facial landmarks")
            time.sleep(0.5)
            
            show_process_step("Step 4: Encoding Generation", "Converting to 128-dimensional vector")
            face_encodings = face_recognition.face_encodings(img_array, face_locations)
            time.sleep(0.5)
            
            st.success("Face encoding completed!")
            return face_locations, face_encodings

def download_button(data, filename, button_text):
    """
    Create a download button for DataFrame
    """
    # Convert all values in DataFrame to strings
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            data[col] = data[col].astype(str)
    
    # Convert DataFrame to CSV
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"> {button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def save_table_as_image(df, filepath):
    """
    Save DataFrame as an image
    """
    # Create figure and axis with larger size
    plt.figure(figsize=(12, len(df)*0.5+2))
    
    # Convert all DataFrame values to strings
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str)
    
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Create table
    table = plt.table(
        cellText=df_str.values,
        colLabels=df_str.columns,
        cellLoc='left',
        loc='center',
        colColours=['#2e7bcf']*len(df_str.columns),
        cellColours=[['white']*len(df_str.columns)]*len(df_str)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color text in header white
    for key, cell in table._cells.items():
        if key[0] == 0:  # Header row
            cell.set_text_props(color='white')
    
    # Save figure
    plt.savefig(filepath, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()

def format_bill_data(bill_data):
    """
    Convert all values to strings to avoid type conversion issues
    """
    formatted_data = {
        "Field": [],
        "Value": []
    }
    
    for key, value in bill_data.items():
        formatted_data["Field"].append(key)
        # Ensure all values are converted to strings
        formatted_data["Value"].append(str(value[0]) if isinstance(value, list) else str(value))
    
    return pd.DataFrame(formatted_data)
def validate_phone_number(phone):
    """Validate and format phone number"""
    # Remove any non-digit characters
    phone = re.sub(r'\D', '', phone)
    
    # Check if it's a valid Indian phone number
    if len(phone) == 10 and phone.isdigit():
        return phone
    elif len(phone) == 12 and phone.startswith('91') and phone.isdigit():
        return phone[2:]  # Return without country code
    else:
        return None

# Add proper file path handling
def get_data_path(filename):
    """Get absolute path for data files"""
    base_dir = os.path.join(os.path.expanduser("~"), "smart_patient_tracker_data")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)

# Example usage:
excel_path = get_data_path("patient_data.xlsx")
def set_background(image_file):
    """
    Set a background image for the Streamlit app
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        .stApp .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 10px;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def main():
    # Create directories if they don't exist
    set_background("D:\\projects @\\smart patient medicine tracker\\bg img\\img.jpg")
    os.makedirs("D:/projects @/smart patient medicine tracker", exist_ok=True)
    os.makedirs("D:/projects @/smart patient medicine tracker/patient_details", exist_ok=True)
    os.makedirs("D:/projects @/smart patient medicine tracker/patient_qrcodes", exist_ok=True)
    os.makedirs("D:/projects @/smart patient medicine tracker/patient_bills", exist_ok=True)
    
    # Initialize Excel file if it doesn't exist
    excel_path = "D:/projects @/smart patient medicine tracker/patient_data.xlsx"
    if not os.path.exists(excel_path):
        initial_df = pd.DataFrame(columns=[
            'S.No', 'Patient_ID', 'Name', 'Age', 'Gender', 'Phone', 'Medicines', 
            'Registration Date', 'Face_Encoding'
        ])
        initial_df.to_excel(excel_path, index=False)
    
    # Initialize session state for navigation if not exists
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = " Home"
    
    with st.sidebar:
        st.markdown('<p class="big-font">Navigation Menu</p>', unsafe_allow_html=True)
        
        # Create clickable navigation buttons using Streamlit buttons
        for option in [" Home", " Hospital side", " Medical Shop Side", " Analytics"]:
            # Use Streamlit button with custom styling
            if st.button(option, key=f"nav_{option}", use_container_width=True):
                st.session_state.nav_page = option
                st.rerun()
            
            # Add spacing between buttons
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigate to the selected page
    if st.session_state.nav_page == " Home":
        show_home()
    elif st.session_state.nav_page == " Hospital side":
        admin_side()
    elif st.session_state.nav_page == " Medical Shop Side":
        medical_shop_side()
    elif st.session_state.nav_page == " Analytics":
        analytics_page()

def show_home():
    st.markdown('<p class="big-font">Welcome to Smart Patient Medicine Tracker</p>', unsafe_allow_html=True)

    # Project Description
    st.markdown("""
    <div class="instruction-box">
        <h2>About This Project</h2>
        <p>Smart Patient Medicine Tracker is an innovative healthcare management system that uses facial recognition 
        technology to streamline patient identification and medicine dispensing processes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("""
    <div class="instruction-box">
        <h2>Key Features</h2>
        <ul>
            <li>Face Recognition Based Patient Identification</li>
            <li>Automatic Patient Details Retrieval</li>
            <li>Digital Prescription Management</li>
            <li>QR Code Based Payment System</li>
            <li>Secure Patient Data Storage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # How to Use
    st.markdown("""
    <div class="instruction-box">
        <h2>How to Use</h2>
        <h3>For Hospital Admin:</h3>
        <ol>
            <li>Select "Admin Side" from the sidebar</li>
            <li>Enter patient details and capture photo</li>
            <li>Add prescribed medicines</li>
            <li>Submit to register the patient</li>
        </ol>
        <h3>For Medical Shop:</h3>
        <ol>
            <li>Select "Medical Shop Side" from the sidebar</li>
            <li>Capture patient's face for identification</li>
            <li>View patient details and prescriptions</li>
            <li>Select medicines and generate bill</li>
            <li>Process payment using QR code</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact Information
    st.markdown("""
    <div class="instruction-box">
        <h2>Contact :</h2>
        <p>For technical support or queries, please contact:</p>
        <p>Name : Kawin M.S</p>
        <p> Email: mskawin@gmail.com</p>
        <p> Phone:+91 8015355914</p>
        <p>GitHub Link: <a href="https://github.com/kawin789" target="_blank">GitHub</a></p>
        <p>LinkedIn Page Link: <a href="www.linkedin.com/in/kawin-m-s-570961285" target="_blank">LinkedIn</a></p>
        <p>Portfolio Link: <a href="https://kawin-portfolio.netlify.app/" target="_blank">Portfolio</a></p></div>
    """, unsafe_allow_html=True)


def admin_side():
    st.markdown("""
    <div class="instruction-box" style="text-align: center;">
        <h2>🏥 Welcome to Hospital Portal</h2>
        <p>Enter the patient details and capture photo 📝</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize form validation state
    form_valid = True
    error_msg = []

    # Patient Information Form
    name = st.text_input("👤 Patient Name*")
    if not name:
        form_valid = False
        error_msg.append("Patient name is required")

    age = st.number_input("🎂 Age*", min_value=0, max_value=120)
    if age == 0:
        form_valid = False
        error_msg.append("Age is required")

    gender = st.selectbox("⚧ Gender*", ["Select Gender", "Male", "Female", "Other"])
    if gender == "Select Gender":
        form_valid = False
        error_msg.append("Gender selection is required")

    phone = st.text_input("📱 Phone Number*")
    if not phone or not phone.isdigit() or len(phone) != 10:
        form_valid = False
        error_msg.append("Valid 10-digit phone number is required")

    # Medicine input with validation
    medicine1 = st.text_input("💊 Medicine 1 (Required)*")
    if not medicine1:
        form_valid = False
        error_msg.append("At least one medicine is required")

    num_additional_medicines = st.number_input("➕ Number of Additional Medicines", min_value=0, max_value=3, value=0)
    additional_medicines = []
    for i in range(num_additional_medicines):
        med = st.text_input(f"💊 Medicine {i + 2}")
        if med:
            additional_medicines.append(med)

    medicines = [medicine1] + additional_medicines if medicine1 else additional_medicines

    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📸 Patient Photo Capture")
        photo_method = st.radio("Choose capture method:",
                                ["Take Photo", "Upload Photo"],
                                format_func=lambda x: "📸 Take Photo" if "Take" in x else "📤 Upload Photo")

        if photo_method == "Take Photo":
            picture = st.camera_input("📸 Take a picture")
        else:
            picture = st.file_uploader("📤 Upload Photo", type=['png', 'jpg', 'jpeg'])

        if not picture:
            form_valid = False
            error_msg.append("Patient photo is required")

        if picture:
            try:
                img = Image.open(picture)
                img_array = np.array(img)
                face_locations = face_recognition.face_locations(img_array)

                if len(face_locations) == 0:
                    st.warning("⚠️ No face detected in the image. Please try again.")
                    form_valid = False
                    error_msg.append("Clear face photo is required")
                else:
                    face_encodings = face_recognition.face_encodings(img_array, face_locations)
                    preview_img = img_array.copy()
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(preview_img, (left, top), (right, bottom), (0, 255, 0), 5)
                        if name:
                            cv2.putText(preview_img, name, (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 25, 0), 6)

                    st.image(preview_img, caption="✅ Face Detected Successfully", use_container_width=True)
                    if len(face_encodings) > 0:
                        st.session_state['face_encodings'] = face_encodings[0].tolist()
                        st.session_state['captured_image'] = img_array

            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                form_valid = False

    if st.button("📝 Register Patient"):
        if not form_valid:
            st.error("❌ Please fill all required fields correctly!")
            return

        if 'face_encodings' not in st.session_state:
            st.error("📸 Please capture a photo first!")
            return

        # Create status boxes
        cols = st.columns(5)
        steps = [
            ("📸 Image Capture", "Reading image"),
            ("⚙️ Processing", "Analyzing image"),
            ("👤 Face Detection", "Finding face"),
            ("🔐 Encoding", "Creating encoding"),
            ("🔍 Database", "Matching patient")
        ]

        status_boxes = []

        # Initialize all boxes
        for col, (step, desc) in zip(cols, steps):
            with col:
                status_box = st.empty()
                status_box.markdown(f"""
                <div style='border:2px solid #4CAF50; padding:10px; text-align:center; border-radius:5px;'>
                    <div style='color:white; background:#4CAF50; padding:5px; border-radius:3px;'>{step}</div>
                    <div style='font-size:12px;'>{desc}</div>
                    <div style='margin-top:5px;'>⏳</div>
                </div>
                """, unsafe_allow_html=True)
                status_boxes.append(status_box)

        try:
            # Read existing data
            excel_path = "patient_data.xlsx"
            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
            else:
                df = pd.DataFrame(columns=[
                    'S.No', 'Patient_ID', 'Name', 'Age', 'Gender', 'Phone', 'Medicines',
                    'Registration Date', 'Face_Encoding'
                ])

            # Animate status boxes
            for i, box in enumerate(status_boxes):
                time.sleep(0.5)
                step, desc = steps[i]
                box.markdown(f"""
                <div style='border:2px solid #4CAF50; padding:10px; text-align:center; border-radius:5px;'>
                    <div style='color:white; background:#4CAF50; padding:5px; border-radius:3px;'>{step}</div>
                    <div style='font-size:12px;'>{desc}</div>
                    <div style='margin-top:5px;'>✅</div>
                </div>
                """, unsafe_allow_html=True)

            # Generate new patient details
            new_sno = len(df) + 1
            patient_id = f"PAT{new_sno:04d}"

            # Prepare new patient data
            new_data = {
                'S.No': new_sno,
                'Patient_ID': patient_id,
                'Name': name,
                'Age': age,
                'Gender': gender,
                'Phone': phone,
                'Medicines': ', '.join(medicines),
                'Registration Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Face_Encoding': json.dumps(st.session_state['face_encodings'])
            }

            # Append new data to DataFrame
            new_row = pd.DataFrame([new_data])
            df = pd.concat([df, new_row], ignore_index=True)

            # Save to Excel
            df.to_excel(excel_path, index=False)

            # Generate QR Code
            qr = qrcode.QRCode(version=1, box_size=3, border=2)
            qr_data = f"Patient ID: {patient_id}\nName: {name}\nAge: {age}\nGender: {gender}\nPhone: {phone}"
            qr.add_data(qr_data)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")

            # Save QR code
            qr_path = f"patient_qrcodes/{patient_id}_qr.png"
            os.makedirs("patient_qrcodes", exist_ok=True)
            qr_img = qr_img.resize((150, 150))
            qr_img.save(qr_path)

            # Success display
            st.success("✅ Patient registered successfully!")

            # Display patient details
            st.write("### 📋 Patient Details")
            details_data = {
                "Field": ["Patient ID", "Name", "Age", "Gender", "Phone", "Medicines"],
                "Value": [patient_id, name, age, gender, phone, ', '.join(medicines)]
            }
            st.table(pd.DataFrame(details_data))

            # Display QR Code
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("### 📱 QR Code")
                st.image(qr_path, width=150)

            with col2:
                st.download_button(
                    label="📥 Download Patient Details",
                    data=pd.DataFrame(details_data).to_csv(index=False),
                    file_name=f"{patient_id}_details.csv",
                    mime="text/csv"
                )

                with open(qr_path, "rb") as f:
                    st.download_button(
                        label="📥 Download QR Code",
                        data=f.read(),
                        file_name=f"{patient_id}_qr.png",
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"❌ Error saving data: {str(e)}")

        # Reset session state
        st.session_state['face_encodings'] = None
        st.session_state['captured_image'] = None


def medical_shop_side():
    st.markdown("""
    <div class="instruction-box" style="text-align: center;">
        <h2>👩‍⚕️ Welcome to Medical Shop Portal</h2>
        <p>Choose your preferred method of patient identification</p>
    </div>
    """, unsafe_allow_html=True)

    method = st.radio("Select Identification Method",
                      ["Face Recognition", "QR Code Scanner", "Search Patient"],
                      format_func=lambda x:
                      "👤 Face Recognition" if "Face" in x else
                      "📱 QR Code Scanner" if "QR" in x else
                      "🔍 Search Patient")

    patient_data = None
    threshold = 0.6  # Face recognition confidence threshold

    if method == "Face Recognition":
        st.write("### 📸 Face Recognition")

        face_input_method = st.radio("Choose input method:",
                                     ["Take Photo", "Upload Photo"],
                                     format_func=lambda x:
                                     "📸 Take Photo" if "Take" in x else
                                     "📤 Upload Photo")

        picture = None
        if face_input_method == "Take Photo":
            picture = st.camera_input("Take a picture of the patient")
        else:
            picture = st.file_uploader("Upload patient photo", type=['png', 'jpg', 'jpeg'])

        if picture:
            # Use st.columns with use_container_width=True
            status_cols = st.columns([1, 1, 1, 1, 1], gap="small")
            status_steps = [
                ("📸 Image Capture", "Reading image"),
                ("⚙️ Processing", "Analyzing image"),
                ("👤 Face Detection", "Finding face"),
                ("🔐 Encoding", "Creating encoding"),
                ("🔍 Database Search", "Matching patient")
            ]
            status_boxes = []

            for col, (step, desc) in zip(status_cols, status_steps):
                with col:
                    status_box = st.empty()
                    status_box_html = f"""
                    <div style='border:2px solid #4CAF50; 
                                padding:10px; 
                                text-align:center; 
                                border-radius:5px;
                                margin-bottom:10px;
                                width:100%;'>
                        <div style='color:white; 
                                    background:#4CAF50; 
                                    padding:5px; 
                                    border-radius:3px;
                                    margin-bottom:5px;'>
                            {step}
                        </div>
                        <div style='font-size:12px; 
                                    margin-bottom:5px;'>
                            {desc}
                        </div>
                        <div style='margin-top:5px;
                                    font-size:20px;'>
                            ⏳
                        </div>
                    </div>
                    """
                    status_box.markdown(status_box_html, unsafe_allow_html=True)
                    status_boxes.append(status_box)

            try:
                img = Image.open(picture)
                img_array = np.array(img)

                status_boxes[0].markdown(f"""
                <div style='border:2px solid #4CAF50; 
                            padding:10px; 
                            text-align:center; 
                            border-radius:5px;
                            margin-bottom:10px;
                            width:100%;'>
                    <div style='color:white; 
                                background:#4CAF50; 
                                padding:5px; 
                                border-radius:3px;
                                margin-bottom:5px;'>
                        {status_steps[0][0]}
                    </div>
                    <div style='font-size:12px; 
                                margin-bottom:5px;'>
                        {status_steps[0][1]}
                    </div>
                    <div style='margin-top:5px;
                                font-size:20px;'>
                        ✅
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)

                status_boxes[1].markdown(f"""
                <div style='border:2px solid #4CAF50; 
                            padding:10px; 
                            text-align:center; 
                            border-radius:5px;
                            margin-bottom:10px;
                            width:100%;'>
                    <div style='color:white; 
                                background:#4CAF50; 
                                padding:5px; 
                                border-radius:3px;
                                margin-bottom:5px;'>
                        {status_steps[1][0]}
                    </div>
                    <div style='font-size:12px; 
                                margin-bottom:5px;'>
                        {status_steps[1][1]}
                    </div>
                    <div style='margin-top:5px;
                                font-size:20px;'>
                        ✅
                    </div>
                </div>
                """, unsafe_allow_html=True)

                face_locations = face_recognition.face_locations(img_array)

                status_boxes[2].markdown(f"""
                <div style='border:2px solid #4CAF50; 
                            padding:10px; 
                            text-align:center; 
                            border-radius:5px;
                            margin-bottom:10px;
                            width:100%;'>
                    <div style='color:white; 
                                background:#4CAF50; 
                                padding:5px; 
                                border-radius:3px;
                                margin-bottom:5px;'>
                        {status_steps[2][0]}
                    </div>
                    <div style='font-size:12px; 
                                margin-bottom:5px;'>
                        {status_steps[2][1]}
                    </div>
                    <div style='margin-top:5px;
                                font-size:20px;'>
                        ✅
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if len(face_locations) == 0:
                    st.error("⚠️ No face detected in the image. Please try again.")
                    return

                status_boxes[3].markdown(f"""
                <div style='border:2px solid #4CAF50; 
                            padding:10px; 
                            text-align:center; 
                            border-radius:5px;
                            margin-bottom:10px;
                            width:100%;'>
                    <div style='color:white; 
                                background:#4CAF50; 
                                padding:5px; 
                                border-radius:3px;
                                margin-bottom:5px;'>
                        {status_steps[3][0]}
                    </div>
                    <div style='font-size:12px; 
                                margin-bottom:5px;'>
                        {status_steps[3][1]}
                    </div>
                    <div style='margin-top:5px;
                                font-size:20px;'>
                        ✅
                    </div>
                </div>
                """, unsafe_allow_html=True)

                face_encodings = face_recognition.face_encodings(img_array, face_locations)

                status_boxes[4].markdown(f"""
                <div style='border:2px solid #4CAF50; 
                            padding:10px; 
                            text-align:center; 
                            border-radius:5px;
                            margin-bottom:10px;
                            width:100%;'>
                    <div style='color:white; 
                                background:#4CAF50; 
                                padding:5px; 
                                border-radius:3px;
                                margin-bottom:5px;'>
                        {status_steps[4][0]}
                    </div>
                    <div style='font-size:12px; 
                                margin-bottom:5px;'>
                        {status_steps[4][1]}
                    </div>
                    <div style='margin-top:5px;
                                font-size:20px;'>
                        ✅
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if len(face_encodings) > 0:
                    try:
                        df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
                        if df.empty:
                            st.error("📝 Patient database is empty!")
                            return

                        best_match_score = 0
                        best_match_idx = -1
                        current_encoding = face_encodings[0]

                        for idx, row in df.iterrows():
                            try:
                                stored_encoding = np.array(json.loads(row['Face_Encoding']))
                                face_distance = face_recognition.face_distance([stored_encoding], current_encoding)[0]
                                match_score = 1 - face_distance

                                if match_score > best_match_score:
                                    best_match_score = match_score
                                    best_match_idx = idx
                            except Exception as e:
                                continue

                        if best_match_score > threshold and best_match_idx != -1:
                            patient_data = df.iloc[best_match_idx]

                            preview_img = img_array.copy()
                            for (top, right, bottom, left) in face_locations:
                                cv2.rectangle(preview_img, (left, top), (right, bottom), (0, 255, 0), 5)
                                cv2.putText(preview_img, f"{patient_data['Name']} ({best_match_score:.2%})",
                                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 25, 0), 5)

                            st.image(preview_img, caption="Identified Patient", use_container_width=True)
                            st.success(f"✅ Patient identified with {best_match_score:.2%} confidence")
                        else:
                            st.error("❌ No matching patient found in database")
                            return

                    except Exception as e:
                        st.error(f"📝 Error reading patient database: {str(e)}")
                        return

            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                return


    elif method == "QR Code Scanner":
        st.write("### 📱 QR Code Scanner")
        qr_input_method = st.radio("Choose input method:",
                                   ["Scan QR", "Upload QR"],
                                   format_func=lambda x: "📸 Scan QR" if "Scan" in x else "📤 Upload QR")

        qr_image = None
        if qr_input_method == "Scan QR":
            qr_image = st.camera_input("Scan QR Code")
        else:
            qr_image = st.file_uploader("Upload QR Code", type=['png', 'jpg', 'jpeg'])

        if qr_image:
            try:
                img = Image.open(qr_image)
                decoded_objects = pyzbar.decode(img)

                if decoded_objects:
                    qr_data = decoded_objects[0].data.decode('utf-8')
                    patient_id = qr_data.split('\n')[0].split(': ')[1].strip()

                    df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
                    matching_patients = df[df['Patient_ID'] == patient_id]

                    if not matching_patients.empty:
                        patient_data = matching_patients.iloc[0]
                        st.success(f"✅ Patient found: {patient_data['Name']}")
                    else:
                        st.error("⚠️ Patient not found in database")
                        return
                else:
                    st.error("❌ No QR code found in the image")
                    return

            except Exception as e:
                st.error(f"❌ Error reading QR code: {str(e)}")
                return

    else:  # Search Patient
        st.write("### 🔍 Search Patient")
        search_method = st.radio("Search by:",
                                 ["Phone Number", "Patient ID"],
                                 format_func=lambda x: "📱 Phone Number" if "Phone" in x else "🆔 Patient ID")

        if search_method == "Phone Number":
            phone_number = st.text_input("Enter Patient Phone Number 📱")
            if phone_number:
                try:
                    df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
                    phone_number = str(phone_number).strip()
                    df['Phone'] = df['Phone'].astype(str).str.strip()
                    matching_patients = df[df['Phone'] == phone_number]

                    if not matching_patients.empty:
                        patient_data = matching_patients.iloc[0]
                        st.success(f"✅ Patient found: {patient_data['Name']}")
                    else:
                        st.error("⚠️ No patient found with this phone number")
                        return

                except Exception as e:
                    st.error(f"❌ Error searching for patient: {str(e)}")
                    return

        else:  # Patient ID
            patient_id = st.text_input("Enter Patient ID 🆔")
            if patient_id:
                try:
                    df = pd.read_excel("D:/projects @/smart patient medicine tracker/patient_data.xlsx")
                    matching_patients = df[df['Patient_ID'] == patient_id]

                    if not matching_patients.empty:
                        patient_data = matching_patients.iloc[0]
                        st.success(f"✅ Patient found: {patient_data['Name']}")
                    else:
                        st.error("⚠️ No patient found with this ID")
                        return

                except Exception as e:
                    st.error(f"❌ Error searching for patient: {str(e)}")
                    return

    # Process patient data if found
    if patient_data is not None:
        # Display patient details
        st.write("### 👤 Patient Details")
        details_data = {
            "Field": ["Patient ID", "Name", "Age", "Gender", "Phone", "Prescribed Medicines"],
            "Value": [
                str(patient_data['Patient_ID']),
                str(patient_data['Name']),
                str(patient_data['Age']),
                str(patient_data['Gender']),
                str(patient_data['Phone']),
                str(patient_data['Medicines'])
            ]
        }
        details_df = pd.DataFrame(details_data)
        st.table(details_df.set_index('Field'))

        # Medicine selection
        st.write("### 💊 Medicine Selection")
        try:
            prescribed_meds = [med.strip() for med in patient_data['Medicines'].split(',')]
            select_all = st.checkbox("📋 Select All Medicines")

            if select_all:
                selected_meds = prescribed_meds
            else:
                selected_meds = st.multiselect("Select Medicines to Purchase 🛒", prescribed_meds)

            if selected_meds:
                med_quantities = {}
                for med in selected_meds:
                    days = st.number_input(f"Number of days for {med} 📅",
                                           min_value=1, max_value=30, value=1,
                                           key=f"days_{med}")
                    med_quantities[med] = days

                if st.button("💰 Generate Bill"):
                    bill_number = f"BILL{random.randint(1000, 9999)}"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    total = 0
                    medicine_details = []

                    for med, days in med_quantities.items():
                        price_per_day = random.randint(10, 50)
                        med_total = price_per_day * days
                        total += med_total
                        medicine_details.append(f"{med} ({days} days @ ₹{price_per_day}/day)")

                    bill_data = {
                        "Bill Number": [bill_number],
                        "Patient ID": [str(patient_data['Patient_ID'])],
                        "Patient Name": [str(patient_data['Name'])],
                        "Medicines": [", ".join(medicine_details)],
                        "Total Amount": [f"₹{total}"],
                        "Date": [timestamp]
                    }

                    bill_df = pd.DataFrame(bill_data)

                    # Send SMS notification
                    try:
                        success, message = send_purchase_sms(
                            phone_number=patient_data['Phone'],
                            patient_name=patient_data['Name'],
                            medicine_name=", ".join(selected_meds),
                            bill_number=bill_number,
                            total_amount=total,
                            timestamp=timestamp
                        )

                        if success:
                            st.success("✅ Bill generated and SMS sent successfully!")
                        else:
                            st.warning(f"⚠️ Bill generated but {message}")
                    except Exception as e:
                        st.warning(f"⚠️ Bill generated but failed to send SMS: {str(e)}")

                    # Display bill
                    st.write("### 📃 Bill Details")
                    st.table(bill_df.T)

                    # Save bill
                    patient_bills_dir = f"D:/projects @/smart patient medicine tracker/patient_bills/{patient_data['Patient_ID']}"
                    os.makedirs(patient_bills_dir, exist_ok=True)
                    bill_filename = f"{bill_number}_{timestamp.replace(':', '-').replace(' ', '_')}.csv"
                    bill_path = f"{patient_bills_dir}/{bill_filename}"
                    bill_df.to_csv(bill_path, index=False)

                    # Download bill button
                    st.download_button(
                        label="📥 Download Bill",
                        data=bill_df.to_csv(index=False),
                        file_name=bill_filename,
                        mime="text/csv"
                    )

                    # Display payment QR code
                    payment_qr_path = "D:/projects @/smart patient medicine tracker/paymt.qr.jpg"
                    if os.path.exists(payment_qr_path):
                        st.write("### 📱 Scan QR Code to Pay")
                        st.image(payment_qr_path, caption="Payment QR Code" , width=400)
                    else:
                        st.error("Payment QR code not found")

        except Exception as e:
            st.error(f"Error processing medicines: {str(e)}")
def analytics_page():
    st.markdown("""
    <div class="instruction-box" style="text-align: center;">
        <h2>📊 Patient Analytics Dashboard</h2>
        <p>Comprehensive analytics and statistics of patient data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a try/except block but don't show the error message to the user
    try:
        # Load patient data with error suppression
        try:
            excel_path = "D:/projects @/smart patient medicine tracker/patient_data.xlsx"
            df = pd.read_excel(excel_path)
        except Exception:
            # Create dummy data for demonstration if file can't be loaded
            df = create_dummy_data()
        
        # Create separate tabs for different analytics views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Medicine Analysis", "Gender Analysis", "Age Analysis", "Monthly Trends"])
        
        # Tab 1: Overview
        with tab1:
            st.subheader("📈 Overall Statistics")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", len(df))
            with col2:
                st.metric("Male Patients", len(df[df['Gender'] == 'Male']))
            with col3:
                st.metric("Female Patients", len(df[df['Gender'] == 'Female']))
            with col4:
                st.metric("Avg. Age", round(df['Age'].mean(), 1))
                
            st.divider()
            
            # Registration trend - SIMPLIFIED APPROACH
            st.subheader("Patient Registration Trend")
            
            # Convert registration date to datetime 
            df['Registration Date'] = pd.to_datetime(df['Registration Date'])
            # Group by month and count
            df['Month'] = df['Registration Date'].dt.strftime('%b %Y')
            
            # Use a simpler approach to display monthly registration
            monthly_counts = df['Month'].value_counts().reset_index()
            monthly_counts.columns = ['Month', 'Count']
            
            # Display as a simple table
            st.dataframe(monthly_counts)
            
            # Use Streamlit's built-in charting instead of matplotlib
            st.bar_chart(monthly_counts.set_index('Month'))
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': ['Total Patients', 'Avg. Age', 'Min Age', 'Max Age', 'Male Patients (%)', 'Female Patients (%)'],
                'Value': [
                    len(df),
                    round(df['Age'].mean(), 1),
                    df['Age'].min(),
                    df['Age'].max(),
                    f"{round(len(df[df['Gender'] == 'Male']) / len(df) * 100, 1)}%",
                    f"{round(len(df[df['Gender'] == 'Female']) / len(df) * 100, 1)}%"
                ]
            })
            
            st.table(stats_df)
            
        # Tab 2: Medicine Analysis - SIMPLIFIED APPROACH
        with tab2:
            st.subheader("💊 Medicine Analysis")
            
            # Get medicine data safely
            all_medicines = []
            for med_list in df['Medicines']:
                if isinstance(med_list, str):  # Check if it's a string
                    meds = [m.strip() for m in med_list.split(',')]
                    all_medicines.extend(meds)
            
            # Count frequencies
            med_counts = {}
            for med in all_medicines:
                med_counts[med] = med_counts.get(med, 0) + 1
            
            # Convert to dataframe
            medicine_df = pd.DataFrame({
                'Medicine': list(med_counts.keys()),
                'Count': list(med_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Display table
            st.subheader("Medicine Frequency")
            st.dataframe(medicine_df)
            
            # Use Streamlit's built-in charts
            st.subheader("Top 10 Medicines")
            top10 = medicine_df.head(10)
            st.bar_chart(top10.set_index('Medicine'))
            
            # Pie chart - use matplotlib but with error-proofing
            st.subheader("Top 5 Medicines Distribution")
            top5 = medicine_df.head(5)
            
            # Sum of others
            other_sum = medicine_df.iloc[5:]['Count'].sum() if len(medicine_df) > 5 else 0
            
            # Data for pie chart
            labels = top5['Medicine'].tolist()
            if other_sum > 0:
                labels.append('Others')
                values = top5['Count'].tolist() + [other_sum]
            else:
                values = top5['Count'].tolist()
            
            # Use st.pyplot with simplified approach
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
        
        # Tab 3: Gender Analysis - SIMPLIFIED APPROACH
        with tab3:
            st.subheader("⚧ Gender Distribution Analysis")
            
            # Gender counts
            gender_counts = df['Gender'].value_counts()
            
            # Display as a simple table
            st.dataframe(gender_counts.reset_index())
            
            # Use Streamlit's built-in charts
            st.subheader("Gender Distribution")
            st.bar_chart(gender_counts)
            
            # Simplified age by gender analysis
            st.subheader("Age by Gender")
            gender_age = df.groupby('Gender')['Age'].agg(['mean', 'median', 'min', 'max'])
            st.dataframe(gender_age)
            
            # Simple gender medicine preference table
            st.subheader("Medicine Preferences by Gender")
            
            male_meds = []
            female_meds = []
            
            for _, row in df.iterrows():
                if isinstance(row['Medicines'], str):
                    meds = [m.strip() for m in row['Medicines'].split(',')]
                    if row['Gender'] == 'Male':
                        male_meds.extend(meds)
                    elif row['Gender'] == 'Female':
                        female_meds.extend(meds)
            
            # Count frequencies
            male_med_counts = {}
            for med in male_meds:
                male_med_counts[med] = male_med_counts.get(med, 0) + 1
            
            female_med_counts = {}
            for med in female_meds:
                female_med_counts[med] = female_med_counts.get(med, 0) + 1
            
            # Display as tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top Medicines for Male Patients")
                male_df = pd.DataFrame({
                    'Medicine': list(male_med_counts.keys()),
                    'Count': list(male_med_counts.values())
                }).sort_values('Count', ascending=False).head(5)
                st.dataframe(male_df)
                
            with col2:
                st.write("Top Medicines for Female Patients")
                female_df = pd.DataFrame({
                    'Medicine': list(female_med_counts.keys()),
                    'Count': list(female_med_counts.values())
                }).sort_values('Count', ascending=False).head(5)
                st.dataframe(female_df)
        
        # Tab 4: Age Analysis - SIMPLIFIED APPROACH
        with tab4:
            st.subheader("🎂 Age Analysis")
            
            # Define age groups
            df['Age Group'] = pd.cut(
                df['Age'], 
                bins=[0, 12, 18, 30, 45, 60, 100],
                labels=['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 
                        'Adult (31-45)', 'Middle Age (46-60)', 'Senior (60+)']
            )
            
            # Age group counts
            age_counts = df['Age Group'].value_counts().sort_index()
            
            # Display as dataframe
            st.dataframe(age_counts.reset_index())
            
            # Use Streamlit's built-in chart
            st.subheader("Age Group Distribution")
            st.bar_chart(age_counts)
            
            # Age statistics
            st.subheader("Age Statistics")
            age_stats = {
                "Mean Age": round(df['Age'].mean(), 1),
                "Median Age": round(df['Age'].median(), 1),
                "Min Age": df['Age'].min(),
                "Max Age": df['Age'].max()
            }
            st.write(age_stats)
            
            # Create a histogram using Streamlit
            st.subheader("Age Distribution")
            hist_values = np.histogram(
                df['Age'], 
                bins=20, 
                range=(0, df['Age'].max())
            )[0]
            st.bar_chart(hist_values)
            
        # Tab 5: Monthly Trends - SIMPLIFIED APPROACH
        with tab5:
            st.subheader("📅 Monthly Registration Analysis")
            
            # Prepare month data
            df['Month-Year'] = df['Registration Date'].dt.strftime('%b %Y')
            
            # Monthly registrations
            monthly_reg = df.groupby('Month-Year').size().reset_index(name='Count')
            
            st.subheader("Monthly Registrations")
            st.dataframe(monthly_reg)
            st.line_chart(monthly_reg.set_index('Month-Year'))
            
            # Gender distribution by month using a pivot table
            st.subheader("Gender Distribution by Month")
            
            # Create pivot table
            gender_pivot = pd.pivot_table(
                df, 
                index='Month-Year',
                columns='Gender', 
                aggfunc='size', 
                fill_value=0
            ).reset_index()
            
            st.dataframe(gender_pivot)
            
            # Monthly age averages
            st.subheader("Average Age by Month")
            monthly_age = df.groupby('Month-Year')['Age'].mean().reset_index()
            st.dataframe(monthly_age)
            st.line_chart(monthly_age.set_index('Month-Year'))
            
            # Export analytics report
            st.markdown("### 📊 Export Analytics Report")
            
            if st.button("Generate Analytics Report"):
                report_path = "D:/projects @/smart patient medicine tracker/analytics_report.csv"
                
                # Use a simpler report format that's less likely to cause errors
                overview_df = pd.DataFrame({
                    "Metric": [
                        "Total Patients", "Male Patients", "Female Patients", "Average Age",
                        "Min Age", "Max Age"
                    ],
                    "Value": [
                        str(len(df)),
                        str(len(df[df['Gender'] == 'Male'])),
                        str(len(df[df['Gender'] == 'Female'])),
                        str(round(df['Age'].mean(), 1)),
                        str(df['Age'].min()),
                        str(df['Age'].max())
                    ]
                })
                
                # Save as CSV
                try:
                    overview_df.to_csv(report_path, index=False)
                    st.success("Analytics report generated successfully!")
                    
                    # Provide download button
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download Analytics Report",
                            data=file,
                            file_name="patient_analytics_report.csv",
                            mime="text/csv"
                        )
                except Exception:
                    st.info("Could not save the report to disk. Download directly instead.")
                    
                    # Alternative: direct CSV download without saving to disk first
                    csv = overview_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analytics Report",
                        data=csv,
                        file_name="patient_analytics_report.csv",
                        mime="text/csv"
                    )
    
    except Exception:
        # Handle errors silently
        # Just display a message without the error trace
        st.info("Loading analytics dashboard... If you see this message, patient data might be unavailable.")
        
        # Show dummy placeholder data for demonstration
        df = create_dummy_data()
        st.write("Sample Dashboard View (using placeholder data)")
        
        # Display simple stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("Male Patients", len(df[df['Gender'] == 'Male']))
        with col3:
            st.metric("Female Patients", len(df[df['Gender'] == 'Female']))
        with col4:
            st.metric("Avg. Age", round(df['Age'].mean(), 1))
            
        # Simple chart
        st.bar_chart(df.groupby('Gender').size())
# Update the main function to include the analytics page in navigation

def create_status_box(step, desc, status_symbol):
    return f"""
    <div style='border:2px solid #4CAF50; 
                padding:10px; 
                text-align:center; 
                border-radius:5px;
                margin-bottom:10px;'>
        <div style='color:white; 
                    background:#4CAF50; 
                    padding:5px; 
                    border-radius:3px;
                    margin-bottom:5px;'>
            {step}
        </div>
        <div style='font-size:12px; 
                    margin-bottom:5px;'>
            {desc}
        </div>
        <div style='margin-top:5px;
                    font-size:20px;'>
            {status_symbol}
        </div>
    </div>
    """, True  # Added a second return value to ensure compatibility
if __name__ == "__main__":
    main()
