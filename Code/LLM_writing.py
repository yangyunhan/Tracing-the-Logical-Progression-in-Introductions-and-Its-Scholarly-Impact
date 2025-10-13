import datetime
import json
from openai import OpenAI
import httpx
import re


class LLMs(object):
    def __init__(self):
        self.path = "Code&data/"
        self.writing_score_path = self.path + "ChatGPT_writing_score/"
        self.intro_path = self.path + "S2ORC_field_introduction/"
        self.s2orc_paper_path = self.path + "S2ORC_field_paper_info/"
        self.goal_key_map = {'Not sure': '0', 'Introduce the topic': '1', 'Describe the background': '2',
                             'Establish the research problem': '3', 'Specify the objectives': '4',
                             'Summarize the findings': '5', 'Map out the paper': '6'}
        self.weight_key_map = {'Strong': '3', 'Moderate': '2', 'Weak': '1'}

    def input_text(self, field):
        paper_dict = {}
        with open(self.s2orc_paper_path + field + '_journal_paper_reference_6paras.txt', 'r') as ftitle:
            for line in ftitle:
                line = line.strip().split(';')
                s2orc_paper_id, year, mag_id, author_num, citation_count, reference_count, title = line
                if mag_id != 'mag_id':
                    paper_dict[s2orc_paper_id] = {'title': title}
        print('{0} field extract title finish!'.format(field))
        processed_paper_dict = {}
        with open(self.writing_score_path + field + '_writing_score.txt', 'r') as fwrite:
            for line in fwrite:
                line = line.strip().split(';')
                s2orc_paper_id = line[0]
                if s2orc_paper_id != 's2orc_paper_id':
                    processed_paper_dict[s2orc_paper_id] = ''
        # new dict
        reprocess_paper_dict = {}
        with open(self.writing_score_path + field + '_.txt', 'r') as fre:
            for line in fre:
                line = line.strip().split(';;;;;')
                s2orc_paper_id = line[0]
                if s2orc_paper_id != 's2orc_paper_id':
                    reprocess_paper_dict[s2orc_paper_id] = ''
        # new dict add end
        with open(self.intro_path + field + '_intro_text.txt', 'r') as fintro, \
                open(self.writing_score_path + field + '_writing_score.txt', 'a') as fws, \
                open(self.writing_score_path + field + '_writing_score_normalization.txt', 'a') as fout:
            fws.write('s2orc_paper_id;writing_score\n')
            fout.write('s2orc_paper_id;len_paragraph;IT;DB;ERQ;SO;SF;MO;IT_0_17;IT_17_34;IT_34_50;IT_50_67;IT_67_84'
                       ';IT_84_100;DB_0_17;DB_17_34;DB_34_50;DB_50_67;DB_67_84;DB_84_100;ERQ_0_17;ERQ_17_34;ERQ_34_50;'
                       'ERQ_50_67;ERQ_67_84;ERQ_84_100;SO_0_17;SO_17_34;SO_34_50;SO_50_67;SO_67_84;SO_84_100;SF_0_17;'
                       'SF_17_34;SF_34_50;SF_50_67;SF_67_84;SF_84_100;MO_0_17;MO_17_34;MO_34_50;MO_50_67;MO_67_84;'
                       'MO_84_100\n')
            i = 0
            for line in fintro:
                s2orc_paper_id, introduction = line.strip().split('-----')
                print('s2orc_paper_id', s2orc_paper_id)
                if s2orc_paper_id != 's2orc_paper_id' and s2orc_paper_id in paper_dict.keys() and \
                        s2orc_paper_id not in processed_paper_dict.keys() and \
                        s2orc_paper_id in reprocess_paper_dict.keys():
                    # 新加的判断条件
                    title = paper_dict[s2orc_paper_id]['title']
                    # 匹配小写字母开头
                    pattern = re.compile(r';;;;;[a-z]')
                    matched_list = pattern.findall(introduction)
                    for item in matched_list:
                        introduction = introduction.replace(item, item[-1])
                    paragraphs = introduction.split(';;;;;')
                    if len(paragraphs) > 5:
                        label_dict = self.test_openapi(s2orc_paper_id, title, paragraphs, field)
                        if label_dict != {}:
                            # ------ gpt按段落写入文件 -------
                            output_ws = ''
                            para_len = len(label_dict.keys())
                            for index in range(1, para_len+1):
                                goal_label = label_dict[index]['goal']
                                weight_label = label_dict[index]['weight']
                                output_ws += '{0}:{1}'.format(goal_label, weight_label)
                                if index < para_len+1:
                                    output_ws += ','
                            fws.write('{0};{1}\n'.format(s2orc_paper_id, output_ws))
                            # ------ gpt按normalization写入文件
                            writing_score_dict = self.process_answer(label_dict)
                            print('writing_score_dict', writing_score_dict)
                            IT_1 = writing_score_dict['1']['0-17']
                            IT_2 = writing_score_dict['1']['17-34']
                            IT_3 = writing_score_dict['1']['34-50']
                            IT_4 = writing_score_dict['1']['50-67']
                            IT_5 = writing_score_dict['1']['67-84']
                            IT_6 = writing_score_dict['1']['84-100']
                            IT = round(IT_1 + IT_2 + IT_3 + IT_4 + IT_5 + IT_6, 2)
                            DB_1 = writing_score_dict['2']['0-17']
                            DB_2 = writing_score_dict['2']['17-34']
                            DB_3 = writing_score_dict['2']['34-50']
                            DB_4 = writing_score_dict['2']['50-67']
                            DB_5 = writing_score_dict['2']['67-84']
                            DB_6 = writing_score_dict['2']['84-100']
                            DB = round(DB_1 + DB_2 + DB_3 + DB_4 + DB_5 + DB_6, 2)
                            ERQ_1 = writing_score_dict['3']['0-17']
                            ERQ_2 = writing_score_dict['3']['17-34']
                            ERQ_3 = writing_score_dict['3']['34-50']
                            ERQ_4 = writing_score_dict['3']['50-67']
                            ERQ_5 = writing_score_dict['3']['67-84']
                            ERQ_6 = writing_score_dict['3']['84-100']
                            ERQ = round(ERQ_1 + ERQ_2 + ERQ_3 + ERQ_4 + ERQ_5 + ERQ_6, 2)
                            SO_1 = writing_score_dict['4']['0-17']
                            SO_2 = writing_score_dict['4']['17-34']
                            SO_3 = writing_score_dict['4']['34-50']
                            SO_4 = writing_score_dict['4']['50-67']
                            SO_5 = writing_score_dict['4']['67-84']
                            SO_6 = writing_score_dict['4']['84-100']
                            SO = round(SO_1 + SO_2 + SO_3 + SO_4 + SO_5 + SO_6, 2)
                            SF_1 = writing_score_dict['5']['0-17']
                            SF_2 = writing_score_dict['5']['17-34']
                            SF_3 = writing_score_dict['5']['34-50']
                            SF_4 = writing_score_dict['5']['50-67']
                            SF_5 = writing_score_dict['5']['67-84']
                            SF_6 = writing_score_dict['5']['84-100']
                            SF = round(SF_1 + SF_2 + SF_3 + SF_4 + SF_5 + SF_6, 2)
                            MO_1 = writing_score_dict['6']['0-17']
                            MO_2 = writing_score_dict['6']['17-34']
                            MO_3 = writing_score_dict['6']['34-50']
                            MO_4 = writing_score_dict['6']['50-67']
                            MO_5 = writing_score_dict['6']['67-84']
                            MO_6 = writing_score_dict['6']['84-100']
                            MO = round(MO_1 + MO_2 + MO_3 + MO_4 + MO_5 + MO_6, 2)
                            fout.write(
                                '{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17};{18};{19};'
                                '{20};{21};{22};{23};{24};{25};{26};{27};{28};{29};{30};{31};{32};{33};{34};{35};{36};{37};'
                                '{38};{39};{40};{41};{42};{43}\n'
                                .format(s2orc_paper_id, len(label_dict.keys()), IT, DB, ERQ, SO, SF, MO, IT_1, IT_2, IT_3, IT_4,
                                        IT_5, IT_6, DB_1, DB_2, DB_3, DB_4, DB_5, DB_6, ERQ_1, ERQ_2, ERQ_3, ERQ_4, ERQ_5,
                                        ERQ_6, SO_1, SO_2, SO_3, SO_4, SO_5, SO_6, SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, MO_1,
                                        MO_2, MO_3, MO_4, MO_5, MO_6))
                i += 1
                if i % 100 == 0:
                    print('{0} field di {1} finish!'.format(field, i))

    def test_openapi(self, s2orc_paper_id, title, paragraphs, field):
        print('now, ask llm')
        prompt_template = f"""
    You are a professional academic reader tasked with labeling paragraphs in the Introduction section of a research paper. 
    Carefully read the title to understand the context of the paper. Then, evaluate the introduction paragraphs based on the authors' goals, which include:
    - Introduce the topic: Clearly communicates the subject's significance, often through an engaging hook.
    - Describe the background: Presents relevant background information or summarizes key research, including major issues and gaps.
    - Establish the research problem: Clarifies the focus of the research, highlighting the significance of the problem and how it differs from previous work.
    - Specify the objectives: Details what the authors aim to discover, including thesis statements or specific hypotheses.
    - Summarize the findings: Presents the outcomes.
    - Map out the paper: Provides an overview of the paper's structure, especially in non-standard formats.
    - Not sure: If a paragraph doesn’t align with these goals or covers multiple goals, classify it as "Not sure."
    Additionally, assign a degree of support for each label:
    - Strong: Clear and compelling evidence.  
    - Moderate: Some evidence, but not fully convincing.  
    - Weak: Little evidence to support the claim.
    For each paragraph, return your assessment in the following format without any other words: 
    [
        {{"paragraph": "..."[:15], "goal": "...", "support": "Strong | Moderate | Weak"}}, 
        ...
    ]
    """
        param_message = [{"role": "system", "content": prompt_template}]
        content = "Now, the title of the paper is: {0}; and paragraphs is: {1}.".format(title, paragraphs)
        content_message = [{"role": "user", "content": content}]
        param_message.extend(content_message)
        client = OpenAI(
            base_url="https://api.xty.app/v1",
            api_key="your_api_key",
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True,
            ),
        )
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=param_message
        )
        # 加入质量控制
        answer = completion.choices[0].message.content
        print('answer', answer)
        try:
            answer_json = json.loads(answer)
        except json.decoder.JSONDecodeError:
            print('catch error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open(self.writing_score_path + field + '_writing_score_check.txt', 'a') as fwrong:
                fwrong.write('s2orc_paper_id-----title-----introductions\n')
                fwrong.write('{0};{1};{2}\n'.format(s2orc_paper_id, title, ';;;;;'.join(paragraphs)))
                return {}
        except KeyError:
            print('catch error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open(self.writing_score_path + field + '_writing_score_check.txt', 'a') as fwrong:
                fwrong.write('s2orc_paper_id-----title-----introductions\n')
                fwrong.write('{0};{1};{2}\n'.format(s2orc_paper_id, title, ';;;;;'.join(paragraphs)))
                return {}
        except AttributeError:
            print('catch error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open(self.writing_score_path + field + '_writing_score_check.txt', 'a') as fwrong:
                fwrong.write('s2orc_paper_id-----title-----introductions\n')
                fwrong.write('{0};{1};{2}\n'.format(s2orc_paper_id, title, ';;;;;'.join(paragraphs)))
                return {}
        else:
            label_dict = {}
            i = 0
            for item in answer_json:
                i += 1
                goal = item['goal']
                weight = item['support']
                if goal.lower() == 'background':
                    goal = 'Describe the background'
                elif goal.lower() == 'topic':
                    goal = 'Introduce the topic'
                elif goal.lower() == 'objectives':
                    goal = 'Specify the objectives'
                elif goal.lower() == 'research problem':
                    goal = 'Establish the research problem'
                goal_score = self.goal_key_map[goal]
                weight_score = self.weight_key_map[weight]
                label_dict[i] = {"goal": goal_score, "weight": weight_score}
            return label_dict

    def process_answer(self, label_dict):
        len_para = len(label_dict.keys())
        writing_score_dict = {
            '0': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
            '1': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
            '2': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
            '3': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
            '4': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
            '5': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
            '6': {'0-17': 0, '17-34': 0, '34-50': 0, '50-67': 0, '67-84': 0, '84-100': 0},
        }
        sum_score = 3 * len_para
        for i in range(1, len_para + 1):
            step_goal = label_dict[i]['goal']
            step_weight = label_dict[i]['weight']
            if 0 < i / len_para < 0.17:
                writing_score_dict[step_goal]['0-17'] += round(int(step_weight) / sum_score, 2)
            elif 0.17 <= i / len_para < 0.34:
                writing_score_dict[step_goal]['17-34'] += round(int(step_weight) / sum_score, 2)
            elif 0.34 <= i / len_para < 0.50:
                writing_score_dict[step_goal]['34-50'] += round(int(step_weight) / sum_score, 2)
            elif 0.50 <= i / len_para < 0.67:
                writing_score_dict[step_goal]['50-67'] += round(int(step_weight) / sum_score, 2)
            elif 0.67 <= i / len_para < 0.84:
                writing_score_dict[step_goal]['67-84'] += round(int(step_weight) / sum_score, 2)
            else:
                writing_score_dict[step_goal]['84-100'] += round(int(step_weight) / sum_score, 2)
        return writing_score_dict

    # 对识别后的writing score进行一次筛选
    def filter_writing_item(self, field, para_len_threshold):
        with open(self.writing_score_path + field + '_writing_score.txt', 'r') as fin, \
                open(self.new_version + field + '_writing_score_' + para_len_threshold + '.txt', 'w') as fout:
            fout.write('s2orc_paper_id;writing_scores\n')
            for line in fin:
                line = line.strip().split(';')
                s2orc_paper_id, scores = line
                if s2orc_paper_id != 's2orc_paper_id':
                    paras = scores[:-1].split(',')
                    zero_count = 0
                    last_para = ''
                    current_len = 0
                    update_writing_score = ''
                    for para in paras:
                        step, weight = para.split(':')
                        if last_para == '6':
                            continue
                        if step == '0':
                            zero_count += 1
                        update_writing_score += '{0}:{1},'.format(step, weight)
                        last_para = step
                        current_len += 1
                    if para_len_threshold == 'full' and int(scores[0]) < 3 and current_len > 5 and zero_count < current_len/2:
                        fout.write('{0};{1}\n'.format(s2orc_paper_id, update_writing_score[:-1]))
                    elif para_len_threshold != 'full' and int(scores[0]) < 3 and int(para_len_threshold) >= current_len > 5 and zero_count < current_len/2:
                        fout.write('{0};{1}\n'.format(s2orc_paper_id, update_writing_score[:-1]))


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    llm = LLMs()
    for field in ['Biology', 'Computer Science', 'Economics', 'Medicine', 'Physics', 'Mathematics']:
        llm.input_text(field)
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done LLMs!')
