import datetime
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd


class Evaluation(object):
    def __init__(self):
        self.steps_gpt = [
            {'201059539': [1,2,2,2,2,4]}, {'9959026': [1,3,2,2,3,4,4,4,4,4,4]},
            {'4879131': [1,2,2,2,2,4]}, {'62937142': [1,2,2,2,2,2,2,2,2,2,2,3,3,3,2,6]},
            {'11073895': [1,1,2,2,3,4,5]}, {'282895': [1,2,2,4,2,2,2,2,2,4,6]},
            {'6786632': [1,2,2,3,4,6]}, {'11439217': [1,2,2,2,2,3,4,6]},
            {'6940862': [1,3,4,2,4,2,2,3,2,4,6]}, {'54901820': [1,2,2,3,4,6]},
            {'154640530': [1,2,4,2,2,2,4,4,2,2,6]}, {'42489969': [1,2,2,2,4,4]},
            {'119317171': [1,2,2,2,2,2,2,2,3,6]}, {'54003942': [2,3,2,2,2,4,6]},
            {'55809208': [2,2,2,2,3,2,2,3,6]}, {'17436911': [1,2,2,3,5,2,6]},
            {'119153874': [1,0,2,4,0,0,0,0,0,2,0,4,0,5,5,5,6]}, {'119607693': [2,3,2,2,4,0,4,4,6]},
            {'17957599': [1,2,3,2,2,4]}, {'57379494': [1,2,2,4,0,6]}, {'34414608': [1,2,2,2,2,4]},
            {'34944152': [1,2,2,2,3,3,3,4,4,4,5,5,6]}, {'859549': [1,2,2,3,4,0,2,2,2,2,5]},
            {'31066390': [1,2,2,2,4,6]}, {'15226132': [2,2,2,2,2,2,2,3,4]},
            {'118438457': [2,3,3,4,2,6]}, {'8274876': [1,2,2,2,4,4,4,4,4,4,5,5,5,2,2,2,6]},
            {'16462794': [2,2,3,4,5,5]}, {'210839077': [1,2,2,2,3,2,6]},
            {'119228630': [1,2,2,2,2,3,2,4,6]},
        ]
        self.clarity_gpt = [
            {'201059539': [3,3,2,2,2,3]}, {'9959026': [3,3,2,3,3,3,2,2,2,2,2]},
            {'4879131': [3,3,3,3,3,3]}, {'62937142': [3,3,2,2,2,2,2,2,2,2,2,3,3,2,2,3]},
            {'11073895': [3,3,3,3,3,3,2]}, {'282895': [3,3,2,3,2,2,2,1,1,3,3]},
            {'6786632': [3,3,2,3,3,3]}, {'11439217': [3,3,3,2,3,3,3,3]},
            {'6940862': [3,3,3,2,3,3,2,3,2,3,3]},
            {'54901820': [3,3,3,3,3,3]}, {'154640530': [3,3,3,2,2,2,1,3,2,2,3]},
            {'42489969': [3,3,3,3,3,3]}, {'119317171': [3,2,3,2,2,2,2,2,3,3]},
            {'54003942': [3,2,3,2,3,3,3]}, {'55809208': [3,2,2,2,3,2,2,3,3]},
            {'17436911': [3,2,3,3,3,2,3]}, {'119153874': [3,1,2,3,1,1,1,1,1,2,1,2,1,3,3,2,2]},
            {'119607693': [3,3,2,2,3,1,3,3,3]}, {'17957599': [3,3,3,2,3,2]},
            {'57379494': [3,3,2,3,1,3]}, {'34414608': [3,2,3,2,2,3]},
            {'34944152': [3,3,3,2,3,3,2,3,2,2,3,3,3]}, {'859549': [3,3,3,3,3,1,2,2,2,2,3]},
            {'31066390': [3,3,2,2,3,3]}, {'15226132': [3,3,3,2,3,2,2,2,3]},
            {'118438457': [3,3,2,3,2,3]}, {'8274876': [3,2,2,2,3,2,2,2,3,2,2,2,2,2,2,2,3]},
            {'16462794': [2,3,3,3,3,3]}, {'210839077': [3,3,3,3,3,2,3]},
            {'119228630': [3,3,3,3,3,3,3,3,3]},
        ]
        self.steps_human = [
            {'201059539': [1, 2, 2, 2, 2, 4]}, {'9959026': [1,2,2,2,3,4,4,4,4,4,4]},
            {'4879131': [1,2,2,2,3,4]}, {'62937142': [1,2,2,2,2,2,2,2,2,2,2,2,3,3,2,6]},
            {'11073895': [1,2,2,2,2,3,5]}, {'282895': [1,2,3,4,2,2,2,2,2,4,6]},
            {'6786632': [1,2,2,3,4,6]}, {'11439217': [1,2,2,2,2,3,3,6]},
            {'6940862': [1,3,4,2,2,4,2,2,3,4,6]},
            {'54901820': [1,2,2,2,4,6]}, {'154640530': [1,2,3,2,2,2,4,4,2,2,6]},
            {'42489969': [1,2,2,2,4,3]}, {'119317171': [1,2,2,2,2,2,2,2,3,6]},
            {'54003942': [2,2,2,2,2,3,6]}, {'55809208': [1,2,2,3,3,2,2,4,6]},
            {'17436911': [1,2,2,3,2,2,6]}, {'119153874': [1,0,2,4,0,0,0,0,0,2,2,0,4,2,5,5,6]},
            {'119607693': [2,2,2,2,2,4,5,5,6]}, {'17957599': [1,2,2,2,2,4]},
            {'57379494': [1,2,2,4,0,6]}, {'34414608': [1,2,2,2,2,4]},
            {'34944152': [1,2,2,2,3,3,3,4,4,4,5,5,6]}, {'859549': [1,2,2,3,4,0,2,2,2,2,5]},
            {'31066390': [2,2,2,2,4,6]}, {'15226132': [1,2,2,2,2,2,2,3,4]},
            {'118438457': [2,3,3,4,2,6]}, {'8274876': [1,2,2,2,4,4,3,4,5,4,4,5,5,2,2,3,6]},
            {'16462794': [2,2,3,4,5,5]}, {'210839077': [1,2,2,2,3,2,6]},
            {'119228630': [1,2,2,2,2,3,2,3,6]},
        ]
        self.clarity_human = [
            {'201059539': [3,3,2,2,3,3]}, {'9959026': [3,3,2,3,3,3,2,2,3,2,2]},
            {'4879131': [3,3,3,3,2,3]}, {'62937142': [3,3,3,2,3,2,2,3,2,3,3,2,3,2,2,3]},
            {'11073895': [3,3,3,3,3,2,2]}, {'282895': [3,3,2,2,2,2,2,2,1,3,3]},
            {'6786632': [3,3,3,3,3,3]}, {'11439217': [3,3,3,2,3,2,3,3]},
            {'6940862': [3,2,3,2,3,2,2,3,2,3,3]},
            {'54901820': [2,3,3,3,3,3]}, {'154640530': [3,3,3,3,2,2,1,3,3,2,3]},
            {'42489969': [3,3,3,3,3,3]}, {'119317171': [3,2,3,3,2,2,3,2,3,3]},
            {'54003942': [3,2,3,2,3,3,3]}, {'55809208': [3,2,2,2,3,2,2,3,3]},
            {'17436911': [3,2,3,3,3,2,3]}, {'119153874': [3,1,2,3,1,1,1,1,1,3,2,2,1,3,3,2,2]},
            {'119607693': [3,2,2,2,3,2,3,3,3]}, {'17957599': [3,3,3,2,3,1]},
            {'57379494': [3,3,3,3,2,3]}, {'34414608': [3,2,3,3,2,3]},
            {'34944152': [3,3,3,2,3,3,2,3,2,2,3,3,3]}, {'859549': [3,3,3,3,3,1,2,3,2,2,3]},
            {'31066390': [3,3,3,3,2,3]}, {'15226132': [3,3,3,2,2,3,2,2,2]},
            {'118438457': [3,3,3,3,2,3]}, {'8274876': [3,2,2,2,3,3,2,2,3,3,2,2,2,3,2,2,3]},
            {'16462794': [2,3,3,3,3,3]}, {'210839077': [3,3,3,3,2,2,3]},
            {'119228630': [3,3,3,3,3,3,3,2,3]},
        ]

    def flatten_annotations(self, steps_machine_list, steps_human_list, clarity_machine_list, clarity_human_list):
        steps_human, steps_machine = [], []
        clarity_human, clarity_machine = [], []
        for idx in range(len(steps_machine_list)):
            # 每篇文章
            article_id_m, machine_steps = list(steps_machine_list[idx].items())[0]
            article_id_h, human_steps = list(steps_human_list[idx].items())[0]
            article_id_cm, machine_clarity = list(clarity_machine_list[idx].items())[0]
            article_id_ch, human_clarity = list(clarity_human_list[idx].items())[0]
            # 确保对应文章 ID 相同
            assert article_id_m == article_id_h == article_id_cm == article_id_ch, "文章ID不一致！"
            # 合并
            steps_machine.extend(machine_steps)
            steps_human.extend(human_steps)
            clarity_machine.extend(machine_clarity)
            clarity_human.extend(human_clarity)
        return steps_human, steps_machine, clarity_human, clarity_machine

    def evaluate_consistency_grouped(self, steps_machine_list, steps_human_list,
                                     clarity_machine_list, clarity_human_list):
        all_steps_human, all_steps_machine = [], []
        all_clarity_human, all_clarity_machine = [], []
        group_results = {}
        n_articles = len(steps_machine_list)
        for idx in range(n_articles):
            article_id_m, machine_steps = list(steps_machine_list[idx].items())[0]
            article_id_h, human_steps = list(steps_human_list[idx].items())[0]
            article_id_cm, machine_clarity = list(clarity_machine_list[idx].items())[0]
            article_id_ch, human_clarity = list(clarity_human_list[idx].items())[0]
            assert article_id_m == article_id_h == article_id_cm == article_id_ch, f"Article ID mismatch at index {idx}"
            n_par = len(human_steps)
            kappa_step = cohen_kappa_score(human_steps, machine_steps) if n_par >= 2 else np.nan
            kappa_clarity = cohen_kappa_score(human_clarity, machine_clarity) if n_par >= 2 else np.nan
            group_results[article_id_m] = {
                'kappa_step': round(kappa_step, 3) if not np.isnan(kappa_step) else np.nan,
                'kappa_clarity': round(kappa_clarity, 3) if not np.isnan(kappa_clarity) else np.nan,
                'n_paragraphs': n_par,
                'steps_human': human_steps,
                'steps_machine': machine_steps,
                'clarity_human': human_clarity,
                'clarity_machine': machine_clarity
            }
            all_steps_human.extend(human_steps)
            all_steps_machine.extend(machine_steps)
            all_clarity_human.extend(human_clarity)
            all_clarity_machine.extend(machine_clarity)
        # 整体结果
        overall_kappa = cohen_kappa_score(all_steps_human, all_steps_machine) if len(all_steps_human) >= 2 else np.nan
        overall_icc = cohen_kappa_score(all_clarity_human, all_clarity_machine) if len(all_steps_human) >= 2 else np.nan
        results = {
            'overall': {'kappa_step': overall_kappa, 'kappa_clarity': overall_icc},
            'by_article': group_results,
            'average': {
                'kappa_step': np.nanmean([v['kappa_step'] for v in group_results.values()]),
                'kappa_clarity': np.nanmean([v['kappa_clarity'] for v in group_results.values()])
            }
        }
        return results

    # 根据 evaluate_consistency_grouped 的结果生成 Excel，每篇文章一行。
    def export_consistency_to_excel(self, results, filename="articles_consistency.xlsx"):
        records = []
        for aid, data in results['by_article'].items():
            records.append({
                'article_id': aid,
                'n_paragraphs': data['n_paragraphs'],
                'steps_human': ','.join(map(str, data['steps_human'])),
                'steps_machine': ','.join(map(str, data['steps_machine'])),
                'clarity_human': ','.join(map(str, data['clarity_human'])),
                'clarity_machine': ','.join(map(str, data['clarity_machine'])),
                'kappa_step': data['kappa_step'],
                'kappa_clarity': data['kappa_clarity']
            })
        df = pd.DataFrame(records)
        df.to_excel(filename, index=False)
        print(f"Excel 文件已保存：{filename}")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    eva = Evaluation()
    steps_human, steps_machine, clarity_human, clarity_machine = eva.flatten_annotations(
        steps_machine_list=eva.steps_gpt,
        steps_human_list=eva.steps_human,
        clarity_machine_list=eva.clarity_gpt,
        clarity_human_list=eva.clarity_human
    )
    # 计算一致性
    results = eva.evaluate_consistency_grouped(
        steps_machine_list=eva.steps_gpt,
        steps_human_list=eva.steps_human,
        clarity_machine_list=eva.clarity_gpt,
        clarity_human_list=eva.clarity_human
    )
    # 输出 Excel，每篇文章一行
    eva.export_consistency_to_excel(results)
    end_time = datetime.datetime.now()
    print(results)
    print(f"It takes {(end_time - start_time).seconds} seconds.")
    print('Done Evaluation!')
