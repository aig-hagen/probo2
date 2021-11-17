import smtplib
import ssl
import email
import os


from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Notification:
    def __init__(self, receiver, subject="Hi there, your experiment is finished",message=""):
        self.port = 465  # For SSL
        self.smtp_server = "smtp.gmail.com"
        self.sender_email = "probonotificationbot@gmail.com"  # Enter your address
        self.receiver_email = receiver  # Enter receiver address
        self.password = "MrRobot123123.probo"
        self.message =  MIMEMultipart()
        self.message["From"] = self.sender_email
        self.message["To"] = self.receiver_email
        self.message["Subject"] = subject
        self.message.attach(MIMEText(message, "plain"))


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
    def attach_files(self, path):
        print(path)
        if os.path.isfile(path):
            print("Attaching file")
            with open(path, "rb") as fil:
               part = MIMEBase("application", "octet-stream")
               part.set_payload(fil.read())
            encoders.encode_base64(part)
        # After the file is closed
            part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(path)}",
                        )
            self.message.attach(part)





