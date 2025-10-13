import os
import datetime
import gzip
import json
import re


class DataCollection(object):
    def __init__(self):
        self.unzip_output = "/MAG_original_data_path/"
        self.original_data_path = self.unzip_output + "20200705v1/full/"
        self.unzip_path = self.original_data_path + "pdf_parses/"
        self.unzip_output_path = self.unzip_output + "S2ORC_json/"
        self.unzip_metadata_path = self.original_data_path + "metadata/"
        self.unzip_metadata_output = self.unzip_path + "meta_files/"

        self.code_data_path = 'Code&data/'
        self.field_full_info = self.code_data_path + "MAG_paper_author_topic/"
        self.intro_path = self.code_data_path + "S2ORC_field_introduction/"

    # read gz file, call unzip_gz_paper to parse gz file to jsonl
    # 解析 meta data 和 文本内容文件 从gz到jsonl文件
    def read_path_gz(self, input_file_name, output_file_name):
        files = os.listdir(input_file_name)
        i = 0
        for file in files:
            if '.gz' in file and '._' not in file:
                self.unzip_gz_paper(file, output_file_name)
                i += 1
            print('di {} file finish!'.format(i))

    # called function, parsing gz files
    def unzip_gz_paper(self, file_name, output_file_name):
        f_name = file_name.replace(".gz", "")
        g_file = gzip.GzipFile(self.unzip_metadata_path + file_name)
        open(output_file_name + f_name, "wb+").write(g_file.read())
        g_file.close()

    # extract papers contain introduction [paper_id, introduction]
    # 获取S2ORC中所有有introduction的 且有至少5个段落的文章 分成10个文件
    def read_file(self, start, end):
        paper_dict = {}
        for j in range(start, end):
            with open(self.unzip_output_path + 'pdf_parses_' + str(j) + '.jsonl', 'r') as fin:
                i = 0
                for line in fin:
                    line_json = json.loads(line)
                    paper_id = line_json['paper_id']
                    body_text = line_json['body_text']
                    introduction_body = []
                    if len(body_text) > 0:
                        for para in body_text:
                            section_title = para['section']
                            if section_title.lower() == 'introduction':
                                introduction_body.append(para['text'])
                    if len(introduction_body) > 4:
                        paper_dict[paper_id] = introduction_body
                    i += 1
                    if i % 10000 == 0:
                        print('di {0} in {1} file'.format(i, j))
        with open(self.unzip_output + 'S2ORC_intro_' + str(start) + '_' + str(end) + '.txt', 'w') as fintro:
            fintro.write('paper_id-----introductions\n')
            m = 0
            for paper_id in paper_dict.keys():
                introductions = ';;;;;'.join(paper_dict[paper_id])
                fintro.write('{0}-----{1}\n'.format(paper_id, introductions))
                m += 1
                if m % 10000 == 0:
                    print('di {0} finish!'.format(m))

    # 找到有introduction的paper的s2orc的paper id
    def search_paper_info(self):
        paper_ids = []
        for file_name in ['0_1', '1_10', '10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90',
                          '90_100']:
            with open(self.unzip_output + 'S2ORC_intro_' + file_name + '_filter.txt', 'r') as fin:
                i = 0
                for line in fin:
                    line = line.strip().split('-----')
                    paper_id, paras = line
                    num_paras = len(paras.split(';;;;;'))
                    if paper_id != 'paper_id' and num_paras > 5:
                        paper_ids.append(paper_id)
                    i += 1
                    if i % 10000 == 0:
                        print('di {0} in {1} finish!'.format(i, file_name))
        with open(self.unzip_output + 'paper_ids_6paras.txt', 'w') as fout:
            j = 0
            for id in paper_ids:
                fout.write('{0}\n'.format(id))
                j += 1
                if j % 10000 == 0:
                    print('di {0} finish!'.format(j))

    # 根据找到的paper id从文章的源信息中（S2ORC）找paper(5 paragraphs)有关的信息
    # s2orc_paper_id;year;mag_id;mag_field_of_study;author_num;mag_multi
    def read_s2orc_paper_mag_paper_id(self):
        s2orc_paper_id_dict = {}
        with open(self.unzip_output + "paper_ids.txt", 'r') as finput:
            for line in finput:
                line = line.strip()
                s2orc_paper_id_dict[line] = {}
        for i in range(0, 100):
            with open(self.unzip_metadata_output + 'metadata_' + str(i) + '.jsonl', 'r') as fin:
                j = 0
                for line in fin:
                    line_json = json.loads(line)
                    paper_id = line_json['paper_id']
                    year = line_json['year']
                    mag_id = line_json['mag_id']
                    mag_field_of_study = line_json['mag_field_of_study']
                    author_num = len(line_json['authors'])
                    if paper_id in s2orc_paper_id_dict.keys() and mag_id != None:
                        s2orc_paper_id_dict[paper_id]['mag_id'] = mag_id
                        s2orc_paper_id_dict[paper_id]['year'] = year
                        s2orc_paper_id_dict[paper_id]['mag_field_of_study'] = mag_field_of_study
                        s2orc_paper_id_dict[paper_id]['author_num'] = author_num
                    j += 1
                    if j % 100000 == 0:
                        print('di {0} in {1} file finish!'.format(j, i))
        with open(self.unzip_output + "paper_mag_info_5paras.txt", 'w') as fout:
            m = 0
            fout.write('s2orc_paper_id;year;mag_id;mag_field_of_study;author_num;mag_multi\n')
            for paper_id in s2orc_paper_id_dict.keys():
                if s2orc_paper_id_dict[paper_id] != {}:
                    year = s2orc_paper_id_dict[paper_id]['year']
                    mag_field_of_study = s2orc_paper_id_dict[paper_id]['mag_field_of_study']
                    author_num = s2orc_paper_id_dict[paper_id]['author_num']
                    mag_id_info = s2orc_paper_id_dict[paper_id]['mag_id']
                    if ',' in mag_id_info:
                        mag_id_list = mag_id_info.split(',')
                        for mag_id in mag_id_list:
                            fout.write('{0};{1};{2};{3};{4};{5}\n'.format(paper_id, year, mag_id, mag_field_of_study,
                                                                          author_num, '1'))
                    else:
                        fout.write('{0};{1};{2};{3};{4};{5}\n'.format(paper_id, year, mag_id, mag_field_of_study,
                                                                      author_num, '0'))
                    m += 1
                    if m % 10000 == 0:
                        print('di {0} finish!'.format(m))

    # 从至少有5段的数据中筛选出有至少六段段文章
    def filter_6paras_s2orc_paper_mag_paper_id(self):
        paper_id_dict = {}
        with open(self.unzip_output + 'paper_ids_6paras.txt', 'r') as fin:
            for line in fin:
                line = line.strip()
                paper_id_dict[line] = {}
        with open(self.unzip_output + 'paper_mag_info_5paras.txt', 'r') as f5para, \
                open(self.unzip_output + 'paper_mag_info_6paras.txt', 'w') as fout:
            fout.write('s2orc_paper_id;year;mag_id;mag_field_of_study;author_num;mag_multi\n')
            for line in f5para:
                lines = line.strip().split(';')
                paper_id = lines[0]
                if paper_id != 's2orc_paper_id' and paper_id in paper_id_dict.keys():
                    fout.write(line)

    # 从获取的introduction文本中做筛选 主要是异常内容的删除
    def filter_method(self):
        paper_dict = {}
        for file_name in ['90_100']:
            with open(self.unzip_output + 'S2ORC_intro_' + file_name + '.txt', 'r') as fin:
                i = 0
                for line in fin:
                    line = line.replace(':;;;;;', ': ')
                    line = line.replace(';;;;;•', '•')
                    line = line.replace(';;;;;(1)', '(1)')
                    line = line.replace(';;;;;(2)', '(2)')
                    line = line.replace(';;;;;(3)', '(3)')
                    line = line.replace('Author(s) agree that this article remain permanently open access under the '
                                        'terms of the Creative Commons Attribution License 4.0 International License',
                                        '')
                    line = line.replace('All rights reserved', '')
                    line = line.replace('E-mail: ', '')
                    line = line.replace('e-mail: ', '')
                    line = line.replace('E-mail address: ', '')
                    line = line.replace('e-mail address: ', '')
                    email_rule = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+'
                    matched_email = re.findall(email_rule, line)
                    if matched_email != []:
                        for e_add in matched_email:
                            line = line.replace(e_add, '')
                    time_rule1 = r'\b(?:\d{1,2}?\s{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]?\s{1,2}(?:20|19)\d{2})\b'
                    time_rule2 = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ](?:\d{1,2})\b[,]?\s{1,2}(?:20|19)\d{2}\b'
                    matched_time1 = re.findall(time_rule1, line)
                    matched_time2 = re.findall(time_rule2, line)
                    matched_time = matched_time1 + matched_time2
                    if matched_time != []:
                        for e_time in matched_time:
                            line = line.replace(e_time, '')
                    if '----We consider ' not in line and 'in this letter' not in line:
                        line = line.strip().split('-----')
                        try:
                            paper_id, paras = line
                        except ValueError:
                            print(line)
                        para_list = paras.split(';;;;;')
                        if len(para_list) > 4:
                            paper_dict[paper_id] = paras
                    i += 1
                    if i % 10000 == 0:
                        print('di {0} in {1} finish!'.format(i, file_name))
            with open(self.unzip_output + 'S2ORC_intro_' + file_name + '_filter.txt', 'w') as fout:
                for paper_id in paper_dict.keys():
                    fout.write('{0}-----{1}\n'.format(paper_id, paper_dict[paper_id]))

    # 筛选出journal s2orc_introduction_journal_paper_id 是根据MAG的source type中来的
    # s2orc_paper_id;year;mag_id;mag_field_of_study;author_num;mag_multi
    def filter_journal_paper_info(self):
        mag_id_dict = {}
        with open(self.unzip_output + 's2orc_introduction_journal_paper_id.txt', 'r') as fjour:
            for line in fjour:
                paper_id = line.strip()
                if paper_id != 'paper_id':
                    mag_id_dict[paper_id] = {}
        with open(self.unzip_output + 'paper_mag_info_6paras.txt', 'r') as fpaper:
            for line in fpaper:
                line = line.strip().split(';')
                s2orc_paper_id, year, mag_id, mag_field_of_study, author_num, mag_multi = line
                if mag_id in mag_id_dict.keys():
                    mag_id_dict[mag_id]['s2orc_paper_id'] = s2orc_paper_id
                    mag_id_dict[mag_id]['year'] = year
                    mag_id_dict[mag_id]['mag_field_of_study'] = mag_field_of_study
                    mag_id_dict[mag_id]['author_num'] = author_num
                    mag_id_dict[mag_id]['mag_multi'] = mag_multi
        with open(self.unzip_output + 'journal_mag_info_6paras.txt', 'w') as fout:
            fout.write('s2orc_paper_id;year;mag_id;mag_field_of_study;author_num;mag_multi\n')
            for mag_id in mag_id_dict.keys():
                fout.write('{0};{1};{2};{3};{4};{5}\n'.format(
                    mag_id_dict[mag_id]['s2orc_paper_id'], mag_id_dict[mag_id]['year'], mag_id,
                    mag_id_dict[mag_id]['mag_field_of_study'], mag_id_dict[mag_id]['author_num'],
                    mag_id_dict[mag_id]['mag_multi']))

    # 根据领域分类 找到每个领域的paper信息 依据这个信息筛选领域
    def filter_fields(self):
        fields_dict = {'Biology': {}, 'Medicine': {}, 'Physics': {}, 'Computer Science': {}, 'Mathematics': {},
                       'Psychology': {}, 'Chemistry': {}, 'Economics': {}, 'Geology': {}, 'Engineering': {},
                       'Political Science': {}, 'Environmental Science': {}, 'Sociology': {}, 'Geography': {},
                       'Materials Science': {}, 'Business': {}, 'Art': {}, 'Philosophy': {}, 'History': {}}
        with open(self.unzip_output + 'journal_mag_info_6paras.txt', 'r') as fin:
            i = 0
            for line in fin:
                line = line.strip().split(';')
                s2orc_paper_id, year, mag_id, mag_field_of_study, author_num, mag_multi = line
                if s2orc_paper_id != 's2orc_paper_id':
                    mag_field_of_study = mag_field_of_study[2:-2]
                    if ',' in mag_field_of_study:
                        mag_fields_list = mag_field_of_study.split('\', \'')
                        for mag_field in mag_fields_list:
                            fields_dict[mag_field][mag_id] = {
                                's2orc_paper_id': s2orc_paper_id, 'year': year, 'author_num': author_num,
                                'mag_multi': mag_multi}
                    else:
                        fields_dict[mag_field_of_study][mag_id] = {
                            's2orc_paper_id': s2orc_paper_id, 'year': year, 'author_num': author_num,
                            'mag_multi': mag_multi}
                i += 1
                if i % 10000:
                    print('di {0} finish!'.format(i))
        for field in fields_dict.keys():
            with open(self.unzip_output + field + '_journal_s2orc_info_6paras.txt', 'w') as fout:
                fout.write('s2orc_paper_id;year;mag_id;author_num;mag_multi\n')
                for mag_id in fields_dict[field].keys():
                    s2orc_paper_id = fields_dict[field][mag_id]['s2orc_paper_id']
                    year = fields_dict[field][mag_id]['year']
                    author_num = fields_dict[field][mag_id]['author_num']
                    mag_multi = fields_dict[field][mag_id]['mag_multi']
                    fout.write('{0};{1};{2};{3};{4}\n'.format(s2orc_paper_id, year, mag_id, author_num, mag_multi))

    # 根据从MAG中匹配到的关于paper author topic相关的信息 选出对应的S2ORC中有introduction文本的paper
    # 获得每个领域的包含MAG中变量的intro_text
    def match_field_intro(self, field):
        paper_intro_dict = {}
        with open(self.field_full_info + field + '_paper_author_topic.txt', 'r') as fin:
            for line in fin:
                line = line.strip().split(';')
                s2orc_paper_id = line[0]
                if s2orc_paper_id != 's2orc_paper_id':
                    paper_intro_dict[s2orc_paper_id] = ''
        print('{0} field first step finish!'.format(field))
        i = 0
        for file_name in ['0_1', '1_10', '10_20', '20_30', '30_40', '40_50', '50_60', '60_70', '70_80', '80_90', '90_100']:
            with open(self.intro_path + 'S2ORC_intro_' + file_name + '_filter.txt', 'r') as fintro:
                j = 0
                for line in fintro:
                    line = line.strip().split('-----')
                    paper_id, introductions = line
                    if paper_id in paper_intro_dict.keys():
                        paper_intro_dict[paper_id] = introductions
                    j += 1
                    if j % 10000 == 0:
                        print('{0} field second step di {1} in {2} file finish!'.format(field, j, i))
                i += 1
        with open(self.intro_path + field + '_intro_text.txt', 'w') as fout:
            fout.write('s2orc_paper_id-----introductions\n')
            for paper_id in paper_intro_dict.keys():
                fout.write('{0}-----{1}\n'.format(paper_id, paper_intro_dict[paper_id]))
        print('{0} field third step finish!'.format(field))


if __name__ == '__main__':
    # 这个code的目的是获取含有两个数据库中所有字段的那些文章的源信息和introduction文本
    # 去MAG中找到并生成paper author topic
    dc = DataCollection()
    start_time = datetime.datetime.now()
    dc.read_path_gz(dc.unzip_metadata_path, dc.unzip_metadata_output)
    dc.search_paper_info()
    dc.read_s2orc_paper_mag_paper_id()
    dc.filter_6paras_s2orc_paper_mag_paper_id()
    dc.filter_journal_paper_info()
    dc.filter_fields()
    dc.read_file(0, 1)
    dc.filter_method()
    for field in ['Biology', 'Computer Science', 'Economics', 'Mathematics', 'Medicine', 'Physics']:
        dc.match_field_intro(field)
    end_time = datetime.datetime.now()
    print('It takes {} minutes.'.format((end_time - start_time).seconds / 60))
    print('Done Data_collection!')
