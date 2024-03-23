import gradio as gr
import numpy as np
import torch
from models.asr_model import ASRModel
from data_control.cmd_data_control import CmdDataControl
from data_control.base_speech_data_control import BaseSpeechDataControl
#from models.vad_model import VADModel





config={
    "recording_sample_rate":44100, # gradio's recording sample rate is 44100
    #vad configuration
    "vad_path":"C:\\test\\asr_integration\\vad",
     "has_speech_threshold":0.6,
     "window_size_samples":512*3,
    "asr_model_path":"C:\\test\\asr_integration\\asr",
   
}
# If we want to design our own record device, VAD is not necessary.
# Because it's much eaiser to detect if there is a voice input on the hardware level.
#vad_model = VADModel(local_path="C:\\test\\silero_vad")
data_control={"CMD":CmdDataControl(config),"SR":BaseSpeechDataControl(config)}

is_stream=False

asr_model=ASRModel(config)
# test 
#no_noise="C:\\test\\asr_integration\\test_data\\no_background_music"
#with_noise="C:\\test\\asr_integration\\test_data\\with_music_background"
#asr_model.predict_file(no_noise)


def audio_file(audio_file:str,new_chunk)->str:
    return asr_model.predict_file(audio_file)
def audio_stream_arrive(audio:np.ndarray, new_chunk) -> str:
    data=audio[1]
    result=data_control[new_chunk].data_arrive(data)
    if result["timeout"]:
        return f'{result["text"]}--->end.'
    return result["text"]
                                    
def predict(audio):
    if isinstance(audio, np.array):
        # predict stream
        return "Error, input is not a valid audio file!"
    elif isinstance(audio,str):
        # predict file
        return "Error, input is not a valid audio file!"
    

# input
inputs = [
    gr.Audio(sources=["microphone"], streaming=True, label='Input audio'),
    gr.Radio(['CMD', 'SR'], label='Language',value='SR')
]
inputs_file=[
    gr.Audio(sources=["microphone"], type="filepath", label='Input audio'),
    gr.Radio(['CMD', 'SR'], label='Language',value='SR')
    ]

output = gr.Textbox(label="Output Text")

text = "实时语音识别Demo"

# description
description = (
    "语音识别Demo,支持关键词唤醒，通用语音识别和无线电系统控制指令识别"  # noqa
)


interface = gr.Interface(
    fn=audio_stream_arrive if is_stream else audio_file,
    inputs=inputs if is_stream else inputs_file,
    outputs=output,
    title=text,
    description=description,
    live=is_stream,
    theme='huggingface',
)
interface.queue()
interface.launch()