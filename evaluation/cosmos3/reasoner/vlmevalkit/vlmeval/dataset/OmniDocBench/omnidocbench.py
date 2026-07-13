import base64
import copy
import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch.distributed as dist
from tqdm import tqdm

from vlmeval.smp import dump, get_intermediate_file_path, load
from ..image_base import ImageBaseDataset

# from ..utils import get_intermediate_file_path, load, dump


class OmniDocBench(ImageBaseDataset):

    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {'OmniDocBench':'https://huggingface.co/datasets/ouyanglinke/OmniDocBench_tsv/resolve/main/OmniDocBench.tsv'}
    DATASET_MD5 = {'OmniDocBench': '0fa5ccf31e682e219cb9ca83da741a59'}


    system_prompt = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

    1. Text Processing:
    - Accurately recognize all text content in the PDF image without guessing or inferring.
    - Convert the recognized text into Markdown format.
    - Maintain the original document structure, including headings, paragraphs, lists, etc.

    2. Mathematical Formula Processing:
    - Convert all mathematical formulas to LaTeX format.
    # - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
    - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

    3. Table Processing:
    - Convert tables to HTML format.
    - Wrap the entire table with <table> and </table>.

    4. Figure Handling:
    - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

    5. Output Format:
    - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
    - For complex layouts, try to maintain the original document's structure and format as closely as possible.

    Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
    '''

    def __init__(self,dataset='OmniDocBench',**kwargs):
        super().__init__(dataset,**kwargs)
        print(f'self.img_root:{self.img_root}')

    def build_prompt(self, line):

        image_path = self.dump_image(line)[0]
        msg = [
            dict(type='image', value=image_path),
            dict(type='text', value=self.system_prompt)
        ]
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        tsv_path=self.data_path
        End2end_evaluator=end2end_evaluator(eval_file,tsv_path)
        Table_evalutor=table_evalutor(eval_file,tsv_path)

        metrics_all=End2end_evaluator.score()
        metircs_table=Table_evalutor.score()

        return metrics_all


class end2end_evaluator():
    def __init__(self,
                 eval_file,
                 tsv_path,
                 match_method:str='quick_match',
                 filter_types:dict=None):
        self.eval_file=eval_file
        self.match_method=match_method
        self.references=[]
        self.predictions = load(eval_file)['prediction'].tolist()
        self.dafault_metircs_dict={
            'text_block':
                {'metric': ['Edit_dist', 'BLEU', 'METEOR']},
            'display_formula':
                {'metric': ['Edit_dist', 'CDM']},
            'table':
                {'metric': ['TEDS', 'Edit_dist']},
            'reading_order':
                {'metric': ['Edit_dist']}
            }

        references = load(tsv_path)['answer'].tolist()

        load_success,load_fail=0,0
        for i,ans in tqdm(enumerate(references),desc='Loading data'):
            try:
                ans = json.loads(ans)
                load_success+=1
                self.references.append(ans) #[{},{}]
            except json.JSONDecodeError as e:
                load_fail+=1
                continue
        print(f'load_success:{load_success},load_fail:{load_fail}')

        filtered_gt_samples = []
        if filter_types:
            for gt_sample in self.references:
                select_flag = True
                for k, v in filter_types.items():
                    if gt_sample["page_info"]["page_attribute"][k] != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = self.references #[{},{},{}]
        self.references=filtered_gt_samples


    def score(self)->dict:
        samples=self.get_matched_elements(self.references,self.predictions)
        metrics=self.process_generated_metric_results(samples)
        return metrics

    def get_page_elements(self, selected_annos):
        saved_element_dict = defaultdict(list)
        related_truncated = []
        truncated_all = {}
        for relation in selected_annos["extra"]["relation"]:   # Handle truncated text issues
            if relation["relation_type"] == 'truncated':
                truncated_all[relation["source_anno_id"]] = ""
                truncated_all[relation["target_anno_id"]] = ""
                exist_flag = False
                for merge_list in related_truncated:
                    if relation["source_anno_id"] in merge_list or relation["target_anno_id"] in merge_list:  # Consider cases where three text blocks may need to be merged
                        merge_list.append(relation["source_anno_id"])
                        merge_list.append(relation["target_anno_id"])
                        exist_flag = True
                if not exist_flag:
                    related_truncated.append([relation["source_anno_id"], relation["target_anno_id"]])

        for item in selected_annos['layout_dets']:
            if item['anno_id'] not in truncated_all.keys():
                saved_element_dict[item["category_type"]].append(item)
            else:
                truncated_all[item['anno_id']] = item

        for merge_list in related_truncated:
            text_block_list = [truncated_all[key] for key in merge_list]
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            text = ""
            for block in sorted_block:
                text += block['text']
            merged_block = {
                "category_type": sorted_block[0]["category_type"], # Directly use information from the first block
                "order": sorted_block[0]["order"],
                "anno_id": sorted_block[0]["anno_id"],
                "text": text,
                "merge_list": sorted_block
            }
            saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)

        return saved_element_dict

    def get_page_elements_list(self, gt_page_elements, category_list):
        element_list = []
        for category_type in category_list:
            if gt_page_elements.get(category_type):
                element_list.extend(gt_page_elements[category_type])
        return element_list

    def get_sorted_text_list(self, selected_annos):
        # txt_type: text, latex, html
        text_list = []
        for item in selected_annos:
            if item.get('order'):
                order = item['order']
            else:
                order = 0
            # 【txt_type,selecte_annos]
            text_list.append((order, item))
        sorted_text_list = sorted(text_list, key=lambda x: x[0])
        return [_[1] for _ in sorted_text_list]

    def filtered_out_ignore(self, items, ignore_category_list):
        filted_items = []
        for item in items:
            if item['gt_category_type'] not in ignore_category_list:
                filted_items.append(item)
        return filted_items

    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position']) for item in order_match_s if (item['gt_position'] != [""] and item['pred_position'] != "")]
        gt_idx_all = [item['gt_position'] for item in order_match_s if (item['gt_position'] != [""])]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]
        read_order_gt = sum(gt_idx_all, []) # Convert to one-dimensional list
        read_order_gt = [x for x in read_order_gt if x]
        gt = sorted(read_order_gt)
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            import Levenshtein
            edit = Levenshtein.distance(gt, pred)/ max(len(pred), len(gt))
            return {
                'gt': gt,
                'pred': pred,
                'img_id': img_name,
                'edit': edit
            }
        else:
            return {}  # If both GT and pred are empty for the page, return empty

    def formula_format(self, formula_matches, img_name):
        # formated_list = []
        for i, item in enumerate(formula_matches):
            item["img_id"] = img_name + '_' + str(i)
        return formula_matches

    def get_matched_elements(self,references:list,predictions:list)->dict:
        from .metrics import recogition_end2end_base_dataset, recogition_end2end_table_dataset

        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []


        for i,sample in enumerate(references):
            img_name = os.path.basename(sample["page_info"]["image_path"])
            pred_content = predictions[i]
            result = self.process_get_matched_elements(sample, pred_content, img_name)
            [plain_text_match_clean, formated_display_formula, latex_table_match_s, html_table_match_s, order_match_single] = result

            if order_match_single:
                order_match.append(order_match_single)
            if plain_text_match_clean:
                plain_text_match.extend(plain_text_match_clean)
            if formated_display_formula:
                display_formula_match.extend(formated_display_formula)
            if latex_table_match_s:
                latex_table_match.extend(latex_table_match_s)
            if html_table_match_s:
                html_table_match.extend(html_table_match_s)

        if len(latex_table_match) > len(html_table_match):
            table_match = latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            table_format = 'html'

        matched_samples_all = {
            "text_block": recogition_end2end_base_dataset(plain_text_match),
            "display_formula": recogition_end2end_base_dataset(display_formula_match),
            "table": recogition_end2end_table_dataset(table_match, table_format),
            "reading_order": recogition_end2end_base_dataset(order_match)
        }

        return matched_samples_all

    def process_get_matched_elements(self, sample, pred_content, img_name):
        from func_timeout import FunctionTimedOut, func_timeout

        from .utils import (match_gt2pred_no_split, match_gt2pred_quick, match_gt2pred_simple,
                            md_tex_filter)

        if self.match_method == 'simple_match':   # add match choice
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'quick_match':
            match_gt2pred = match_gt2pred_quick
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            # print('Invalid match method name. The quick_match will be used.')
            match_gt2pred = match_gt2pred_quick

        pred_dataset = md_tex_filter(pred_content)
        gt_page_elements = self.get_page_elements(sample)

        text_all = self.get_page_elements_list(gt_page_elements, ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                                                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                                                'header', 'footer', 'page_footnote', 'page_number'])


        display_formula_match_s = []
        plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        order_match_single = []
        if text_all:
            gt_text_list = self.get_sorted_text_list(text_all)
            try:
                plain_text_match_s = func_timeout(
                    30, match_gt2pred, args=(gt_text_list, pred_dataset['text_all'], 'text', img_name)
                )
            except FunctionTimedOut as e1:
                print(f'Time out for plain text match of {img_name}, match_gt2pred_simple will be used.')
                plain_text_match_s = match_gt2pred_simple(gt_text_list, pred_dataset['text_all'], 'text', img_name)
            except Exception as e:
                print(str(e))
                sys.exit()

            if not plain_text_match_s:
                print(f'No text match of {img_name}. The plain text match will be empty.')
            else:
                plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption'])


        if gt_page_elements.get('equation_isolated'):
            gt_display_list = self.get_sorted_text_list(gt_page_elements['equation_isolated'])
            display_formula_match_s = match_gt2pred(gt_display_list, pred_dataset['equation_isolated'], 'formula', img_name)
            display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != [""]]
            if not display_formula_match_s:
                print(f'No display_formula_match of {img_name}. The display_formula_match will be empty.')

        if gt_page_elements.get('table'):
            gt_table_list = self.get_sorted_text_list(gt_page_elements['table'])
            if pred_dataset['latex_table']:
                latex_table_match_s = match_gt2pred_simple(gt_table_list, pred_dataset['latex_table'], 'latex_table', img_name)
                latex_table_match_s = [x for x in latex_table_match_s if x['gt_idx'] != [""]]
            if pred_dataset['html_table']:
                html_table_match_s = match_gt2pred_simple(gt_table_list, pred_dataset['html_table'], 'html_table', img_name)
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]
            else:
                html_table_match_s = match_gt2pred_simple(gt_table_list, [], 'html_table', img_name)
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]


        order_match_s = plain_text_match_clean
        if order_match_s:
            order_match_single = self.get_order_paired(order_match_s, img_name)

        return [plain_text_match_clean, display_formula_match_s, latex_table_match_s, html_table_match_s, order_match_single]

    def process_generated_metric_results(self,samples,save_name:str='end2end_quick_match'):
        from .metrics import METRIC_REGISTRY, get_full_labels_results, get_page_split, show_result

        result_all={}
        page_info={}
        metircs_dict=self.dafault_metircs_dict
        pages=self.references #gt_samples list

        for page in pages:
                img_path=os.path.basename(page['page_info']['image_path'])
                page_info[img_path]=page['page_info']['page_attribute']

        for element in metircs_dict.keys():

            result={}
            group_info=metircs_dict[element].get('group',[])
            # samples = samples.get(element) ##
            cur_samples = samples[element]

            for metric in metircs_dict[element]['metric']:
                metric_val = METRIC_REGISTRY.get(metric)

                cur_samples,result_s = metric_val(cur_samples).evaluate(group_info, f"{save_name}_{element}")
                if result_s:
                    result.update(result_s)

            if result:
                print(f"{element}")
                show_result(result)
            result_all[element]={}


            group_result=get_full_labels_results(cur_samples)
            page_result=get_page_split(cur_samples,page_info)

            result_all[element]={
                'all':result,
                'group':group_result,
                'page':page_result
            }
            if isinstance(cur_samples,list):
                saved_samples=cur_samples
            else:
                saved_samples=cur_samples.samples
            # NOTE: The original code has a bug here, it will overwrite the result file in each iteration.
            # I will fix it by adding element to the filename.
            # NOTE: Fixed typo .josn -> .json
            result_file = get_intermediate_file_path(self.eval_file, f'_{save_name}_{element}_result', 'json')
            dump(saved_samples, result_file)

        metric_result_file = get_intermediate_file_path(self.eval_file, f'_{save_name}_metric_result', 'json')
        dump(result_all, metric_result_file)

        dict_list = []
        save_dict={}
        en_overall=[]
        ch_overall=[]
        for category_type, metric in [("text_block", "Edit_dist"), ("display_formula", "Edit_dist"), ("display_formula", "CDM"), ("table", "TEDS"), ("table", "Edit_dist"), ("reading_order", "Edit_dist")]:
            if metric == 'CDM':
                save_dict[category_type+'_'+metric+'_EN'] = '-'
                save_dict[category_type+'_'+metric+'_CH'] = '-'
            elif metric == "TEDS":
                save_dict[category_type+'_'+metric+'_EN'] = result_all[category_type]["page"][metric]["language: english"] * 100
                save_dict[category_type+'_'+metric+'_CH'] = result_all[category_type]["page"][metric]["language: simplified_chinese"] * 100
            else:
                save_dict[category_type+'_'+metric+'_EN'] = result_all[category_type]["page"][metric].get("language: english", np.nan)
                save_dict[category_type+'_'+metric+'_CH'] = result_all[category_type]["page"][metric].get("language: simplified_chinese",np.nan)
            if metric == "Edit_dist":
                en_overall.append(result_all[category_type]["page"][metric].get("language: english", np.nan))
                ch_overall.append(result_all[category_type]["page"][metric].get("language: simplified_chinese",np.nan))

        save_dict['overall_EN'] = sum(en_overall) / len(en_overall)
        save_dict['overall_CH'] = sum(ch_overall) / len(ch_overall)
        dict_list.append(save_dict)
        df = pd.DataFrame(dict_list,index=['end2end',]).round(3)

        e2e_eval_file = get_intermediate_file_path(self.eval_file, '_End2End_Evaluation', 'json')
        dump(result_all, e2e_eval_file)
        
        overall_file = get_intermediate_file_path(self.eval_file, '_overall')
        dump(df, overall_file)

        print(f"The save path of End2End_Evaluation is: {e2e_eval_file}")
        print(f"The save path of overall metrics is: {overall_file}")
        return df


class table_evalutor():
    def __init__(self,eval_file,tsv_path):
        self.eval_file = eval_file
        gt_key='html'
        pred_key='pred'
        self.category_filter='table'
        self.category_type='table'
        self.metircs_list=['TEDS','Edit_dist']
        self.gt_samples,self.table_samples=self.load_data(eval_file,tsv_path,pred_key,gt_key)

    def load_data(self,eval_file,gt_file,pred_key,gt_key):
        from .data_preprocess import (clean_string, normalized_formula, normalized_table,
                                      textblock2unicode)
        samples=[]
        preds=[]
        predictions=load(eval_file)['prediction'].tolist()
        gt_samples=load(gt_file)['answer'].tolist()
        load_success,load_fail=0,0
        for i,gt_sample in tqdm(enumerate(gt_samples),desc='Loading data'):
            try:
                ans=json.loads(gt_sample)
                for item in ans['layout_dets']:
                    if item['category_type']=="table":
                        item['pred']=predictions[i]
                        load_success+=1
                        preds.append(ans)

            except json.JSONDecodeError as e:
                load_fail+=1
                continue
        print(f'load_table_success:{load_success},load_table_fail:{load_fail}')

        count=0
        for pred in preds:
            img_name = os.path.basename(pred['page_info']['image_path'])
            for i, ann in enumerate(pred['layout_dets']):
                if not ann.get(gt_key):
                    continue
                if self.category_filter:
                    if ann['category_type'] not in self.category_filter:
                        continue
                if not ann.get(pred_key):
                    # print(f'Cannot find pred for {img_name}. ann is {ann}')
                    # pdb.set_trace()
                    count += 1
                    continue
                else:
                    gt_text = ann[gt_key]
                    norm_gt = gt_text
                    pred_text = ann[pred_key]
                    norm_pred = pred_text
                    if self.category_type:
                        if self.category_type == 'text':
                            norm_gt = clean_string(textblock2unicode(ann[gt_key]))
                            norm_pred = clean_string(textblock2unicode(ann[pred_key]))
                        elif self.category_type == 'formula':
                            norm_gt = normalized_formula(ann[gt_key])
                            norm_pred = normalized_formula(ann[pred_key])
                        elif self.category_type == 'table':
                            norm_gt = normalized_table(ann[gt_key], gt_key)
                            norm_pred = normalized_table(ann[pred_key], gt_key)
                        else:
                            raise ValueError(f'Invalid category type: {self.category_type}')

                samples.append({
                    "gt": gt_text,
                    "norm_gt": norm_gt,
                    "gt_attribute": [ann['attribute']],
                    'pred': pred_text,
                    "norm_pred": norm_pred,
                    'img_id': img_name
                })

        print(f'Cannot find pred for {count} samples.')
        return preds,samples

    def score(self)->dict:
        metrics=self.process_generated_metric_results()
        return metrics

    def process_generated_metric_results(self,save_name:str='OmniDocBench_table'):
        from .metrics import METRIC_REGISTRY, get_full_labels_results, get_page_split, show_result

        p_scores={}
        page_info={}
        no_page_flag=False
        samples=self.table_samples
        pages=self.gt_samples

        for page in pages:
            if 'page_info' not in page:
                no_page_flag=True
                break
            img_path=os.path.basename(page['page_info']['image_path'])
            page_info[img_path]=page['page_info']['page_attribute']

        for metric in self.metircs_list:
            metric_val=METRIC_REGISTRY.get(metric)
            samples, result = metric_val(samples).evaluate({}, save_name)
            if result:
                p_scores.update(result)
        show_result(p_scores)
        group_result=get_full_labels_results(samples)
        if no_page_flag:
            page_result={}
        else:
            page_result=get_page_split(samples,page_info)

        result_all={
            'all':p_scores,
            'group':group_result,
            'page':page_result
        }

        metric_result_file = get_intermediate_file_path(self.eval_file, f'_{save_name}_metric_result', 'json')
        dump(result_all, metric_result_file)

        dict_list=[]
        dict_list.append(result_all["group"]["TEDS"])

        df4 = pd.DataFrame(dict_list, index=['OmniDocBench_table'])
        df4 = df4 * 100
        df4 = df4.round(1)
        selected_columns = df4[["language: table_en", "language: table_simplified_chinese", "language: table_en_ch_mixed", "line: full_line", "line: less_line", "line: fewer_line", "line: wireless_line",
                        "with_span: True", "with_span: False", "include_equation: True", "include_equation: False", "include_background: True", "include_background: False", "table_layout: vertical", "table_layout: horizontal"]]

        table_attr_file = get_intermediate_file_path(self.eval_file, '_table_attribute')
        dump(selected_columns, table_attr_file)
        print(f'The save path of table_attribute is :{table_attr_file}')
        return selected_columns
