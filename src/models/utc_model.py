
from paddlenlp import Taskflow

class UTCModel:
    def __init__(self, config: dict):
        task_path=config.get("task_path", None)
        assert(task_path is not None)
        schema=config.get("schema", [])
        home_path=config.get("home_path", None)
        task_name=config.get("task_name", "zero_shot_text_classification")
        pre_trained_model=config.get("pre_trained_model", "utc-base")
        self.task=Taskflow(task=task_name,model=pre_trained_model,schema=schema,task_path=task_path,precision="fp16",home_path=home_path)

    def predict(self, text: str) -> dict:
        result=self.task(text)
        if len(result)==0 or len(result[0]["predictions"])==0:
            return {"text":"","probability":0}
        return {"text":result[0]["predictions"][0]["label"],"probability":result[0]["predictions"][0]["score"]}

