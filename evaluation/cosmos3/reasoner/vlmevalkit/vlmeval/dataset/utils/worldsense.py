import re

import numpy as np

from vlmeval.smp import load
from .multiple_choice import extract_answer_from_item

FAIL_MSG = 'Failed to obtain answer via API.'

DURATIONS = [
    "<1min",
    "1-2min",
    "2-4min",
    "4-6min",
    "6-8min",
    ">8min"
]

DOMAINS = [
    'Tech & Science',
    'Culture & Politics',
    'Daily Life',
    'Film & TV',
    'Performance',
    'Games',
    'Sports',
    'Music',
]

SUB_CATEGORIES = [
    "Academic Lectures",
    "Auto",
    "Software",
    "Physics",
    "Climate Change",
    "Space Missions",
    "Chemistry",
    "Engineering Projects",
    "Biology",
    "Science Explainers",
    "Artificial Intelligence",
    "Astronomy",
    "Tech Reviews",
    "Editorials",
    "Politics",
    "Historical Analysis",
    "Social Commentary",
    "Book Reviews",
    "Cultural Explainers",
    "Drawing Tutorials",
    "Celebrity Interviews",
    "Art Exhibitions",
    "Fashion",
    "Travel",
    "Daily Vlogs",
    "Cooking",
    "Pranks",
    "Camping",
    "Nutrition & Health",
    "Home Improvement",
    "Painting & Photography",
    "Unboxing Videos",
    "Family Vlogs",
    "DIY & Crafts",
    "Skincare & Makeup",
    "Documentaries",
    "Film Trailers",
    "Event Livestreams",
    "Short Films",
    "Documentary Profiles",
    "Movie Reviews",
    "World News",
    "Talks",
    "Parodies",
    "Storytime",
    "Stand-up",
    "Sketches",
    "FPS Game",
    "Casual Game",
    "Role Playing Game",
    "Sports Game",
    "Basketball",
    "Racing",
    "Football",
    "Bowling Ball",
    "Soccer",
    "Motorsport",
    "swimming",
    "Boxing",
    "Other Sports",
    "Fitness",
    "Fishing",
    "Hiking",
    "Covers",
    "Music Videos",
    "Remixes",
    "Walkthroughs"
]

TASK_DOMAINS = [
    'Recognition',
    'Understanding',
    'Reasoning'
]

TASK_CATEGORIES = [
    "Anomaly Recognition",
    "Event Recognition",
    "Attribute Recognition",
    "Human Interaction",
    "Temporal Localization",
    "Video Emotions",
    "Event Sorting",
    "Hallucination",
    "Text and Diagram Understanding",
    "Attribute Reasoning",
    "Causal Reasoning",
    "Object Counting",
    "Action Counting",
    "Temporal Prediction",
    "Emotion Change",
    "Audio Counting",
    "Scene Recognition",
    "Human-object Interaction",
    "Human Emotions",
    "Object State Change",
    "Relation Reasoning",
    "Spatial Relation",
    "Audio Source Localization",
    "Audio Recognition",
    "Object Existence Recognition",
    "Audio Change"
]

AUDIO_CLASSES = [
    "Speech",
    "Event",
    "Music",
]


def get_dimension_rating(data_path):
    data = load(data_path)

    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ['overall']:
        duration_rating[duration] = {
            'overall': '',
            'domain': {k: [] for k in DOMAINS},
            'sub_category': {k: [] for k in SUB_CATEGORIES},
            'task_domain': {k: [] for k in TASK_DOMAINS},
            'task_type': {k: [] for k in TASK_CATEGORIES},
            'audio_class': {k: [] for k in AUDIO_CLASSES},
        }

    for i in range(len(data)):

        domain = data.iloc[i]['domain']
        sub_ctg = data.iloc[i]['sub_category']
        task_domain_ctg = data.iloc[i]['task_domain']
        task_ctg = data.iloc[i]['task_type']
        audio_ctg = eval(data.iloc[i]['audio_class'])

        duration = data.iloc[i]['duration']
        score = float(data.iloc[i]['score'])

        duration_rating['overall']['domain'][domain].append(score)
        duration_rating['overall']['sub_category'][sub_ctg].append(score)
        duration_rating['overall']['task_domain'][task_domain_ctg].append(score)
        duration_rating['overall']['task_type'][task_ctg].append(score)

        duration_rating[duration]['domain'][domain].append(score)
        duration_rating[duration]['sub_category'][sub_ctg].append(score)
        duration_rating[duration]['task_domain'][task_domain_ctg].append(score)
        duration_rating[duration]['task_type'][task_ctg].append(score)

        for _audio_ctg in audio_ctg:
            duration_rating['overall']['audio_class'][_audio_ctg].append(score)
            duration_rating[duration]['audio_class'][_audio_ctg].append(score)

    for duration in ['overall'] + DURATIONS:

        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.3f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.3f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['sub_category'][sub_ctg] = sub_res_dur

        for task_ctg in TASK_DOMAINS:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_domain"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_domain'][task_ctg] = task_res_dur

        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

        for audio_ctg in AUDIO_CLASSES:
            audio_res_dur = f'{np.mean([x for x in duration_rating[duration]["audio_class"][audio_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['audio_class'][audio_ctg] = audio_res_dur

    return duration_rating


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
