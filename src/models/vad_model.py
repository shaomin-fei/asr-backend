import torch
import numpy as np

class VADModel:
    def __init__(self,local_path:str,**config) -> None:
        USE_ONNX = False
        if config is None:
            config={}
        if local_path:
            self.model,self.utils=torch.hub.load(repo_or_dir=local_path,
                              model='silero_vad',
                              source='local',
                              force_reload=True,
                              onnx=USE_ONNX)
        else:
            self.model,self.utils=torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)
        self.has_speech_threshold=config.get("has_speech_threshold", 0.6)
        self.sample_rate=config.get("recording_sample_rate", 16000)
        self.window_size_samples=config.get("window_size_samples", 512*3)
            
    def has_speech(self,audio) -> bool:
        '''
        detect if the audio stream has speech.
        return true if any chunk of the audio stream has speech.
        '''
        for i in range(0, len(audio), self.window_size_samples):
            audio_chunk = audio[i:i + self.window_size_samples]
            if len(audio_chunk) < self.window_size_samples:
                audio_chunk=audio[-1*self.window_size_samples:]
            prob=self.model(audio_chunk,self.sample_rate).item()
            #print(f"prob={prob}")
            if prob>self.has_speech_threshold:
                #print(f"Has audio i={i},prob={prob}")
                return True
        return False
        #prob=self.model(audio,self.sample_rate).item()
        #print(f"prob={prob}")
        #if prob>self.has_speech_threshold:
        #    return True 
        #return False

