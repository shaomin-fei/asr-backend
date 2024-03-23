from models.asr_model import ASRModel


config={
    "recording_sample_rate":44100, # gradio's recording sample rate is 44100
    #vad configuration
    "vad_path":"C:\\test\\asr_integration\\vad",
     "has_speech_threshold":0.6,
     "window_size_samples":512*3,
    "model_path":"C:\\test\\asr_integration\\asr",
   
}


is_stream=False

asr_model=ASRModel(config)
# test 
no_noise="C:\\test\\asr_integration\\test_data\\no_background_music.wav"
with_noise="C:\\test\\asr_integration\\test_data\\with_music_background.wav"
monitor_g="C:\\test\\asr_integration\\test_data\\monitor-g.wav" # good
pcm_8k="C:\\test\\asr_integration\\test_data\\8k_pcm.wav"
result=asr_model.predict_file(pcm_8k)
print(f"result text={result['text']},confidence={result['confidence']}")