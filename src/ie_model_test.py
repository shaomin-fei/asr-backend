from models.ie.predictor import UIEPredictor


config={
    "recording_sample_rate":44100, # gradio's recording sample rate is 44100
    #vad configuration
    "vad_path":"C:\\test\\asr_integration\\vad",
    "has_speech_threshold":0.6,
    "window_size_samples":512*3,
    "asr_model_path":"C:\\test\\asr_integration\\asr",
    "ref_path":"C:\\test\\asr_integration\\wakeup\\samples",
    "ie":{
    "model_path":"C:\\test\\asr_integration\\ie\\best.ckpt",
    "pretrained_model":"C:\\test\\asr_integration\\ie",
    "pretrained_tokenizer":"C:\\test\\asr_integration\\ie",
    "device":"cpu",
    "positon_prob":0.5,
    "max_seq_len":512,
    "batch_size":63,
    "split_sentence":False,
    "schema":["监测站","省监测中心","任务","市监测中心","中心频率","带宽","起始频率","截止频率","步进","设备","解调方式","声音开关","自动增益","频段扫描类别"]
    }
    
}
str12="东林市熊猫大道监测站做600到900兆的DSCAN，步进300k。"

ie_model=UIEPredictor(config["ie"])
result=ie_model(str12)