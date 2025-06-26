import smtplib
from email.message import EmailMessage
import datetime

if __name__ == "__main__":

  s = smtplib.SMTP('mail.cse.cuhk.edu.hk')

  msg = EmailMessage()

  msg['Subject'] = 'Experiment Status'
  msg['From'] = 'Chunxiao Ye <cxye23@cse.cuhk.edu.hk>'
  msg['To'] = 'chunxy@link.cuhk.edu.hk'

  msg.set_content(\
f"""The experiment has finished running at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

Best,
Chunxiao""")

  s.send_message(msg)

  s.quit()
