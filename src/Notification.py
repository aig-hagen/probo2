import smtplib
import ssl
import email

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Notification:
    def __init__(self, receiver):
        self.port = 465  # For SSL
        self.smtp_server = "smtp.gmail.com"
        self.sender_email = "probonotificationbot@gmail.com"  # Enter your address
        self.receiver_email = receiver  # Enter receiver address
        self.password = "MrRobot123123.probo"
        self.message =  MIMEMultipart()
        self.message["From"] = self.sender_email
        self.message["To"] = self.receiver_email
        self.message["Subject"] = "Hi there, your experiment is finished"
        #self.message.attach(MIMEText("Enclosed you will find a report of your experiment.", "plain"))


    def send(self):
        context = ssl.create_default_context()
        text = self.message.as_string()
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, self.receiver_email, text)
    def add_attachment(self,file):
        # Open PDF file in binary mode
        with open(file, "rb") as attachment:
         # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {file}",
                        )

        # Add attachment to message and convert message to string
        self.message.attach(part)
    
