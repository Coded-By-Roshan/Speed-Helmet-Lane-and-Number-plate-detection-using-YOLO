import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_violation_email(receiver_email, driver_name, vehicle_number, violation_type):
    sender_email = "" 
    sender_password = "" 
    subject = "Notice of Traffic Rule Violation – {}".format(violation_type)
    message = f"""
    Dear {driver_name},

    We hope this email finds you well. This is to inform you that a violation of traffic rules was recorded under your vehicle **{vehicle_number}\n.
    As per the traffic regulations, you are required to settle the penalty within time to avoid further legal actions. Payment can be made through the online portal or the nearest traffic office.
    Failure to comply may result in additional fines or legal consequences. If you believe this notice has been issued in error, you may file an appeal by visiting the nearest traffic office within 7 days.

    For further assistance, please contact us at traffic@nepalpolice.gov.np.

    Your cooperation in maintaining road safety is highly appreciated.

    Best regards,  
    **Nepal Traffic Police**  
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()

        print("Email sent successfully to", receiver_email)

    except Exception as e:
        print("Failed to send email:", str(e))
