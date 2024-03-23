from models.vad_model import VADModel

# test VADModel
# the path can't have chinese characters
#local_path = "C:\\test\\silero_vad"
#model = VADModel(local_path=local_path)

import pyttsx3
engine=pyttsx3.init()
voices = engine.getProperty('voices')
filtered_voices=list(filter(lambda x:'Chinese' in x.name,voices))
if filtered_voices is not None and len(filtered_voices)>0:
    voice=filtered_voices[0]
else:
    voice=voices[0]
engine.setProperty('voice', voice.id)
engine.say("设备DDF550")
engine.runAndWait()
