from gtts import gTTS 
  
# This module is imported so that we can  
# play the converted audio 
import os 
   
mytext = 'Welcome to the ASL translator'
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False) 
myobj.save("welcome.mp3") 
os.system("afplay welcome.mp3") 