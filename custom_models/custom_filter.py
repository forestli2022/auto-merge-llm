import collections
import re
import sys
import unicodedata

from lm_eval.filters.extraction import Filter, RegexFilter
from lm_eval.api.registry import register_filter

import re
import sys
import jsonlines
from fraction import Fraction

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion, is_zh=False, is_ja=False):
    if is_zh:
        text = completion.split('答案是：')
    elif is_ja:
        text = completion.split('答案是：')
    else:   
        text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

@register_filter("get_answer")
class GetAnswer(Filter):
    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            for resp in r:
                filtered = extract_answer_number(resp)
                if filtered == None:
                    filtered_resps.append("[invalid]")
                else:
                    filtered_resps.append(str(filtered))
        return filtered_resps

@register_filter("get_answer_zh")
class GetAnswerZh(Filter):
    def apply(self, resps, docs):
        filtered_resps = []
        
        for r, doc in zip(resps, docs):
            for resp in r:
                filtered = extract_answer_number(resp, is_zh=True)
                if filtered == None:
                    filtered_resps.append("[invalid]")
                else:
                    filtered_resps.append(str(filtered))
        return filtered_resps

@register_filter("get_answer_ja")
class GetAnswerJa(Filter):
    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            for resp in r:
                filtered = extract_answer_number(resp, is_ja=True)
                if filtered == None:
                    filtered_resps.append("[invalid]")
                else:
                    filtered_resps.append(str(filtered))
        return filtered_resps

@register_filter("get_code")
class GetCode(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)