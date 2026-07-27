import re

from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item
from vlmeval.smp.file import load

FAIL_MSG = 'Failed to obtain answer via API.'

DOMAINS = [
    'Character Motivation Causality',
    'Element Counting',
    'Event Counting',
    'Event Localization',
    'Element Localization',
    'Positional Relationship',
    'Objective Causality (Videography Phenomenon & Illusion)',
    'Plot Attribute',
    'Plot Attribute (Montage)',
    'Element Attributes (Optical Illusion)',
    'Element Attributes',
    'Event Duration & Speed Attribute',
    'Objective Causality',
    'Character Reaction Causality',
    'Professional Knowledge',
    'Displacement Attribute',
    'Character Emotion Attribute',
    'Local Event Attribute'
]


def get_dimension_rating(data_path):
    data = load(data_path)

    domain_rating = {k: {} for k in DOMAINS}
    for domain in DOMAINS + ['overall']:
        domain_rating[domain] = {
            'number': 0,
            'correct': 0,
            'score': 0.0,
        }

    for i in range(len(data)):

        domain = data.iloc[i]['capability']
        score = data.iloc[i]['score']

        domain_rating['overall']['number'] += 1
        domain_rating[domain]['number'] += 1
        if score > 0:
            domain_rating['overall']['correct'] += 1
            domain_rating[domain]['correct'] += 1

    for domain in DOMAINS + ['overall']:
        domain_rating[domain]['score'] = domain_rating[domain]["correct"] / domain_rating[domain]["number"] * 100

    return domain_rating


def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]
