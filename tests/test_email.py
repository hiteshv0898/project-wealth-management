import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def send_test_email():
    try:
        msg = EmailMessage()
        msg.set_content("This is a test email from your Wealth Management Agent.")
        msg['Subject'] = "Test Email - Wealth Management Agent"
        msg['From'] = os.getenv("EMAIL_SENDER")
        msg['To'] = os.getenv("EMAIL_RECEIVER")

        print("✅ Sending test email...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
            smtp.send_message(msg)
        print("📧 Test email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Run the test
send_test_email()
