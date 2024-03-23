class SpeechLifeTime():
    def __init__(self, speech_lifetime_second=1):
        self.speech_lifetime_second = speech_lifetime_second
        self.speech_start = False
        self.speech_end = True
        self.data_len=0

    def is_end(self,data_len, sample_rate):
        self.data_len+=data_len
        if self.data_len>self.speech_lifetime_second*sample_rate:
            self.speech_end=True
            self.data_len=0
            return True
        return False
    
    def is_start(self):
        return self.speech_start
    
    def start(self):
        self.speech_start=True
        self.speech_end=False
        self.data_len=0

    def reset(self):
        self.speech_start=False
        self.speech_end=True
        self.data_len=0

    def set_lifetime(self, speech_lifetime_second):
        self.speech_lifetime_second = speech_lifetime_second