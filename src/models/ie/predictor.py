import math
import re
import json

import torch
import numpy as np
from transformers import AutoConfig

from models.ie.UIELightningPredictMoudle import UIELightningPredictMoudle
from models.ie.SpanEvaluator import get_bool_ids_greater_than, get_span
from models.ie.UIELightningMoudle import UIELightningMoudle
from models.ie.utils import dbc2sbc, cut_chinese_sent, get_id_and_prob


class SchemaTree(object):
    """
    Implementataion of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)


class PytorchEngine(object):
    def __init__(self, model,device):
        self.model = model
        self.device=device
        self.model.eval()
    def infer(self, input_dict):
        import torch
        for input_name, input_value in input_dict.items():
            input_value = torch.LongTensor(input_value)
            if self.device == 'gpu':
                input_value = input_value.cuda()
            input_dict[input_name] = input_value
        with torch.no_grad():
            outputs = self.model(**input_dict)
        start_prob, end_prob = outputs[0], outputs[1]
        if self.device == 'gpu':
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
        start_prob = start_prob.detach().numpy()
        end_prob = end_prob.detach().numpy()
        return start_prob, end_prob


class UIEPredictor(object):
    def __init__(self, config:dict):
        from models.ie.UIEModel import UIE
        from transformers import AutoTokenizer
        model_path=config.get("model_path","../models")
        schema=config.get("schema",[])
        pretrained_model=config.get("pretrained_model","../models")
        pretrained_tokenizer=config.get("pretrained_tokenizer","../models")
        # with open(f"{pretrained_tokenizer}/config.json",encoding="utf-8") as reader:
        #     pretrained_config=json.load(reader)
        device=config.get("device","cpu")
        positon_prob=config.get("positon_prob",0.5)
        max_seq_len=config.get("max_seq_len",512)
        batch_size=config.get("batch_size",63)
        split_sentence=config.get("split_sentence",False)

        # self.encoder = UIELightningMoudle.load_from_checkpoint(model_path,model_path=pretrained_model,lr=1e5)
        self.encoder=UIELightningPredictMoudle.load_from_checkpoint(checkpoint_path=model_path,pretrained_config_path=pretrained_tokenizer)
        self.encoder.eval()
        # if it's a .bin file, we don't have to save the info to variable 'ckpt', we can load it directly
        self.engine = PytorchEngine(self.encoder,device=device)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        self._schema_tree = None
        self.set_schema(schema)
        self._max_seq_len=max_seq_len
        self._batch_size=batch_size
        self._split_sentence=split_sentence
        self._position_prob=positon_prob
        self._multilingual=False
        self._is_en=False

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self.build_tree(schema)

    def build_tree(self, schema, name="root"):
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError("Invalid schema, value for each key:value pairs should be list or string"
                                        "but {} received".format(type(v)))
                    schema_tree.add_child(self.build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree

    def __call__(self, inputs):
        texts=inputs
        if isinstance(texts,str):
            texts=[texts]
        results=self._multi_stage_predict(texts)
        # by default, probility is the np float32, which is not json serializable
        # so we need to convert it to float
        for result in results:
            for k,v in result.items():
                for i in range(len(v)):
                    v[i]["probability"]=float(v[i]["probability"])
        return results

    def _multi_stage_predict(self, datas):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            datas (list): a list of strings
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `datas`
        """
        results = [{} for _ in range(len(datas))]
        # input check to early return
        if len(datas) < 1 or self._schema_tree is None:
            return results

        # copy to stay `self._schema_tree` unchanged
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for data in datas:
                    examples.append({
                        "text": data,
                        "prompt": dbc2sbc(node.name)
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, data in zip(node.prefix, datas):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find(
                                        "[", 1)].strip()
                                    cls_options = re.search(
                                        r'\[.*?\]$', node.name).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append({
                                "text": data,
                                "prompt": dbc2sbc(prompt)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(datas))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relations"].keys(
                        ):
                            relations[k][i]["relations"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[v[i]])

                new_relations = [[] for i in range(len(datas))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys(
                        ) and node.name in relations[i][j]["relations"].keys():
                            for k in range(
                                    len(relations[i][j]["relations"][
                                            node.name])):
                                new_relations[i].append(relations[i][j][
                                                            "relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(datas))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self._is_en:
                            prefix[k].append(" of " +
                                             result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {"text": prompt[start:end],
                              "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
        '''
        Split the raw texts automatically for model inference.
        Args:
            input_texts (List[str]): input raw texts.
            max_text_len (int): cutting length.
            split_sentence (bool): If True, sentence-level split will be performed.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
            input_mapping (dict): mapping between raw text and short input texts.
        '''
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for text in input_texts:
            if not split_sentence:
                sens = [text]
            else:
                sens = cut_chinese_sent(text)
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = [cnt_short]
                    else:
                        input_mapping[cnt_org].append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = temp_text_id
                    else:
                        input_mapping[cnt_org].extend(temp_text_id)
            cnt_org += 1
        return short_input_texts, input_mapping

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3

        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        sentence_ids = []
        probs = []

        input_ids = []
        token_type_ids = []
        attention_mask = []
        offset_maps = []

        if self._multilingual:
            padding_type = "max_length"
        else:
            padding_type = "longest"
        encoded_inputs = self._tokenizer(
            text=short_texts_prompts,
            text_pair=short_input_texts,
            # stride=2,
            truncation=True,
            max_length=self._max_seq_len,
            padding=padding_type,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np")

        start_prob_concat, end_prob_concat = [], []
        for batch_start in range(0, len(short_input_texts), self._batch_size):
            input_ids = encoded_inputs["input_ids"][batch_start:batch_start + self._batch_size]
            token_type_ids = encoded_inputs["token_type_ids"][batch_start:batch_start + self._batch_size]
            attention_mask = encoded_inputs["attention_mask"][batch_start:batch_start + self._batch_size]
            offset_maps = encoded_inputs["offset_mapping"][batch_start:batch_start + self._batch_size]
            if self._multilingual:
                input_ids = np.array(
                    input_ids, dtype="int64")
                attention_mask = np.array(
                    attention_mask, dtype="int64")
                position_ids = (np.cumsum(np.ones_like(input_ids), axis=1)
                                - np.ones_like(input_ids)) * attention_mask
                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids
                }
            else:
                input_dict = {
                    "input_ids": np.array(
                        input_ids, dtype="int64"),
                    "token_type_ids": np.array(
                        token_type_ids, dtype="int64"),
                    "attention_mask": np.array(
                        attention_mask, dtype="int64")
                }

            outputs = self.engine.infer(input_dict)
            start_prob, end_prob = outputs[0], outputs[1]
            start_prob_concat.append(start_prob)
            end_prob_concat.append(end_prob)
        start_prob_concat = np.concatenate(start_prob_concat)
        end_prob_concat = np.concatenate(end_prob_concat)

        start_ids_list = get_bool_ids_greater_than(
            start_prob_concat, limit=self._position_prob, return_prob=True)
        end_ids_list = get_bool_ids_greater_than(
            end_prob_concat, limit=self._position_prob, return_prob=True)

        input_ids = input_dict['input_ids']
        sentence_ids = []
        probs = []
        for start_ids, end_ids, ids, offset_map in zip(start_ids_list,
                                                       end_ids_list,
                                                       input_ids.tolist(),
                                                       offset_maps):
            for i in reversed(range(len(ids))):
                if ids[i] != 0:
                    ids = ids[:i]
                    break
            span_list = get_span(start_ids, end_ids, with_prob=True)
            # fsm update
            # prompt_length = len(short_inputs[0]["prompt"])
            # prompt_len_plus_special_token=prompt_length+2
            probs=[]
            # sentence_id, prob = get_id_and_prob(span_list, offset_map.tolist())

            # for start, end in span_list:
            #     prob = start[1] * end[1]
            #     sentence_ids.append([(start[0]-prompt_len_plus_special_token,end[0]-prompt_len_plus_special_token)])
            #     probs.append([prob])
            # fsm commented out
            sentence_id, prob = get_id_and_prob(span_list, offset_map.tolist())
            sentence_ids.append(sentence_id)
            probs.append(prob)

        # fsm
        # if len(sentence_ids)==0:
        #     sentence_ids.append([])
        #     probs.append([])
        ###
        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts,
                                    self.input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0][
                            'text']] = [1, short_results[v][0]['probability']]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text': cls_res,
                        'probability': cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def predict(self, input_data):
        results = self._multi_stage_predict(input_data)
        return results

