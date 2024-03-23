import numpy as np

def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    probs=np.array(probs)
    result = []
    dim_len=len(probs.shape)
    if dim_len>1:
        for p in probs:
            result.append(get_bool_ids_greater_than(p,limit,return_prob))
        return result
    else:
        for i, p in enumerate(probs):
            if p>limit:
                if return_prob:
                    result.append((i,p))
                else:
                    result.append(i)
        return result


def eval_span(pre_start_ids, pred_end_ids, label_start_ids, label_end_ids):
    pred_set=get_span(pre_start_ids,pred_end_ids)
    label_set=get_span(label_start_ids,label_end_ids)
    num_correct=len(pred_set&label_set)
    num_infer=len(pred_set)
    num_label=len(label_set)
    return (num_correct,num_infer,num_label)


class SpanEvaluator:
    def __init__(self):
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def compute(self,start_pos,end_pos,gold_start_ids,gold_end_ids):
        pred_start_ids=get_bool_ids_greater_than(start_pos.to("cpu"))
        pred_end_ids=get_bool_ids_greater_than(end_pos.to("cpu"))
        gold_start_ids=get_bool_ids_greater_than(gold_start_ids.tolist())
        gold_end_ids=get_bool_ids_greater_than(gold_end_ids.tolist())
        num_correct_spans=0
        num_infer_spans=0
        num_label_spans=0
        for pred_start_ids,pred_end_ids,label_start_ids,label_end_ids in zip(
            pred_start_ids,pred_end_ids,gold_start_ids,gold_end_ids
        ):
            [_correct,_infer,_label]= eval_span(pred_start_ids, pred_end_ids, label_start_ids, label_end_ids)
            num_correct_spans+=_correct
            num_infer_spans+=_infer
            num_label_spans+=_label
        return num_correct_spans,num_infer_spans,num_label_spans

    def update(self,num_correct_spans, num_infer_spans, num_label_spans):
        self.num_infer_spans=num_infer_spans
        self.num_correct_spans=num_correct_spans
        self.num_label_spans=num_label_spans

    def accumulate(self):
        precision=float(self.num_correct_spans/self.num_infer_spans) if self.num_infer_spans else 0
        recall=float(self.num_correct_spans/self.num_label_spans) if self.num_label_spans else 0
        f1_score=float(2*precision*recall/(precision+recall)) if self.num_correct_spans else 0
        return precision,recall,f1_score

    def reset(self):
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0
