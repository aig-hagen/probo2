import smtplib
import ssl
import email
import os


from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Notification:
    def __init__(self, receiver, subject="Hi there, your experiment is finished",message="",id=None):
        self.port = 465  # For SSL
        self.smtp_server = "smtp.gmail.com"
        self.sender_email = "probonotificationbot@gmail.com"  # Enter your address
        self.receiver_email = receiver  # Enter receiver address
        self.password = "MrRobot123123.probo"
        self.message =  MIMEMultipart()
        self.message["From"] = self.sender_email
        self.message["To"] = self.receiver_email
        self.message["Subject"] = subject
        self.foot = "Note: Since the access data for this e-mail account are public, please do not open any attachments to e-mails in which the identification code does not match the one generated for you."
        self.id = id
        self.message.attach(MIMEText(message, "plain"))
        self.message.attach(MIMEText(self.foot, "plain"))
        self.message.attach(MIMEText(f'ID: {str(self.id)}', "plain"))



    def send(self):
        context = ssl.create_default_context()
        text = self.message.as_string()
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as server:
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, self.receiver_email, text)
        except Exception:
            print("Notification e-mail could not be sent due to issues with credentials.")

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
    def attach_file(self, path):

        if os.path.isfile(path):
            print(f"Attaching file {path} to e-mail.")
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
        else:
            print(f"File {path} not found")

    def attach_mutiple_files(self,path):
        if isinstance(path,list):
            for f in path:
                self.attach_file(f)
        else:
            for f in os.listdir(path):
                self.attach_file(f)





