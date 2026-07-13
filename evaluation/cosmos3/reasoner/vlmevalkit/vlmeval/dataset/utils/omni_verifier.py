import re

import numpy as np

from .judge_util import build_judge
from .yorn import YOrN_Extraction

EVAL_TMPL = """
You are an AI assistant tasked with evaluating whether a model's response correctly answers
a given visual-language question.
You will be provided with: 1. The model's response; 2. The ground truth answer.
Your task is to determine whether the model's response conveys the same meaning as the ground truth.
The response is considered **correct** if:
- It has the same meaning as the ground truth, even if phrased differently.
- It provides additional relevant details without altering the original meaning.
The response is considered **wrong** if:
- It contradicts the ground-truth
- It misses essential information or include additional incorrect information.
Your evaluation should include a single word (Either `"yes"` (if correct) or `"no"` (if incorrect)).

Now please complete the following task:
[Begin Response]{response}[End Response]
[Begin Ground-Truth]{ground_truth}[End Ground-Truth]
"""


def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0', 'zero': '0',
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
        'nineteen': '19', 'twenty': '20',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


class OmniVerifier:

    tmpl_map = {
        'boxed': r'\boxed\{([^}]*)\}',
        'brace': r'\{([^}]*)\}'
    }

    def __init__(self,
                 tmpl=None,
                 judge='gpt-4o',
                 lower_case=True,
                 rule_only=False,
                 retry=3,
                 timeout=60,
                 **kwargs):

        self.judge = build_judge(model=judge, retry=retry, timeout=timeout)
        self.tmpl = tmpl if tmpl not in self.tmpl_map else self.tmpl_map[tmpl]
        self.lower_case = lower_case
        self.rule_only = rule_only
        if retry >= 3:
            self.t_series = list(np.arange(0, 1, 1 / (retry - 1))) + [1.0, ]
        else:
            self.t_series = [0, 1] if retry == 2 else [0, ]
        self.match = -1

    def verify(self, prediction, gt, **kwargs):
        prediction = str(prediction)
        gt = str(gt)
        if self.tmpl is not None:
            matches = re.findall(self.tmpl, prediction)
            if len(matches):
                if isinstance(self.match, int) and self.match < len(matches):
                    prediction = matches[self.match]
        if self.lower_case:
            prediction = prediction.lower()
            gt = gt.lower()
        prediction = _process_digit_article(prediction)
        gt = _process_digit_article(gt)
        if gt == prediction:
            return True, 'Rule Match'
        else:
            if self.rule_only:
                return False, f'Rule Match: Prediction: {prediction}, GT: {gt}'
            judge_prompt = EVAL_TMPL.format(response=prediction, ground_truth=gt)
            for t in self.t_series:
                res = self.judge.generate(judge_prompt, temperature=t, **kwargs)
                answer = YOrN_Extraction(res)
                if answer in ['Yes', 'No']:
                    return answer == 'Yes', 'Judge Match'
        return False, f'Judge Failed: Prediction: {prediction}, GT: {gt}'
