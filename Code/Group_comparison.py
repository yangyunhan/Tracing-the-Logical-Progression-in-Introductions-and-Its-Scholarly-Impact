import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import pingouin as pg


class GroupComparison(object):
    def __init__(self):
        self.common_path = "Code&data/"
        self.writing_score = self.common_path + "ChatGPT_writing_score/"
        self.output = self.common_path + 'Analysis_output/'
        self.field_para_90 = {
            'Biology': 10,
            'Computer Science': 12,
            'Economics': 13,
            'Medicine': 10,
            'Physics': 12,
            'Psychology': 12,
            'Mathematics': 14,
            'Geology': 10
        }
        self.citation_90 = {
            'Biology': 115,
            'Computer Science': 87,
            'Economics': 146,
            'Medicine': 114,
            'Physics': 100,
            'Psychology': 121,
            'Mathematics': 57,
            'Geology': 137
        }
        self.citation_80 = {
            'Biology': 65,
            'Computer Science': 31,
            'Economics': 74,
            'Medicine': 63,
            'Physics': 55,
            'Psychology': 62,
            'Mathematics': 22,
            'Geology': 85
        }
        self.citation_10 = {
            'Biology': 1,
            'Computer Science': 0,
            'Economics': 1,
            'Medicine': 0,
            'Physics': 1,
            'Psychology': 0,
            'Mathematics': 0,
            'Geology': 2
        }
        self.citation_20 = {
            'Biology': 3,
            'Computer Science': 2,
            'Economics': 3,
            'Medicine': 2,
            'Physics': 3,
            'Psychology': 2,
            'Mathematics': 1,
            'Geology': 6
        }

    def group(self, field, input_file="_regression.txt", output_file="_group_10.txt"):
        df = pd.read_csv(self.output + field + input_file, sep=';')
        upper_val = self.citation_90[field]
        lower_val = self.citation_10[field]

        def group_func(x, upper_val=upper_val, lower_val=lower_val):
            if x <= lower_val:
                return 0
            elif x >= upper_val:
                return 1
            else:
                return 2

        df['citation_group'] = df['citation_count'].apply(group_func)
        df_extreme = df[df['citation_group'].isin([0, 1])].copy()
        # 保存结果
        df_extreme.to_csv(self.output + field + output_file, sep=';', index=False)
        print(f"field {field} 分组完成，总共保留 {len(df_extreme)} 条数据（组0和1）")
        print(df_extreme[['s2orc_paper_id', 'citation_count', 'citation_group']].head())

    def run_psm(self, field, caliper=0.05):
        print(f"now is field {field}")
        pd.set_option('display.max_columns', None)
        df = pd.read_csv(self.output + field + '_group_10.txt', sep=';')
        treatment_col = 'citation_group'
        match_vars = ['year', 'author_num', 'topic_size', 'first_author_prod', 'first_author_h_index',
                      'last_author_prod', 'last_author_h_index', 'title_len', 'len_paragraph', 'journal_cc']
        # 1. 标准化匹配变量
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[match_vars])
        # 2. 构建倾向性得分模型
        model = LogisticRegression()
        model.fit(X_scaled, df[treatment_col])
        # 3. 计算倾向性得分
        df['propensity_score'] = model.predict_proba(X_scaled)[:, 1]
        # 4. 拆分组
        treated = df[df[treatment_col] == 1].copy()
        print('group == 1', len(treated))
        control = df[df[treatment_col] == 0].copy()
        print('group == 0', len(control))
        # 5. 匹配：为每个treated找最近邻control(1:1匹配)
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(control[['propensity_score']])
        distances, indices = nbrs.kneighbors(treated[['propensity_score']])
        matched_indices = []
        for i, dist in enumerate(distances):
            if dist[0] <= caliper:
                treated_idx = treated.index[i]
                control_idx = control.index[indices[i][0]]  # 允许重复
                matched_indices.append((treated_idx, control_idx))
        # 6. 返回匹配结果
        matched_df = pd.DataFrame(matched_indices, columns=['treated_idx', 'control_idx'])
        print('len is ', len(matched_indices))
        treated_matches = df.loc[matched_df['treated_idx']].copy()
        control_matches = df.loc[matched_df['control_idx']].copy()
        treated_matches['match_group'] = 'treated'
        control_matches['match_group'] = 'control'
        matched_data = pd.concat([treated_matches, control_matches])
        # 7. 保存结果为TXT文件
        matched_data.set_index('s2orc_paper_id', inplace=True)
        matched_data.to_csv(self.output + field + '_psm_10.txt', sep=';', encoding='utf-8')
        print(f"匹配结果已保存")

    # 对指定字段的数据，比较treated和control组每个变量的均值差异，并输出到Excel
    def two_group_ttest_to_excel(self, field):
        pd.set_option('display.max_columns', None)
        df = pd.read_csv(self.output + field + '_psm.txt', sep=';')
        compare_cols = [
            'IT', 'DB', 'ERQ', 'SO', 'SF', 'MO',
            'IT_0_17', 'DB_17_34', 'DB_34_50', 'DB_50_67',
            'ERQ_50_67', 'ERQ_67_84', 'SO_50_67', 'SO_67_84', 'SO_84_100',
            'SF_84_100', 'MO_84_100',
            'coverage_weighted', 'spearman_seq', 'unclear_ratio'
        ]
        high_citation = df.loc[df['match_group'] == 'treated']
        low_citation = df.loc[df['match_group'] == 'control']
        results = []
        for col in compare_cols:
            g1 = pd.to_numeric(high_citation[col], errors='coerce').dropna()
            g2 = pd.to_numeric(low_citation[col], errors='coerce').dropna()
            if len(g1) == 0 or len(g2) == 0:
                print(f"⚠️ Column {col} has no valid data in one group, skipping.")
                continue
            avg1 = g1.mean()
            avg2 = g2.mean()
            ttest_res = pg.ttest(g1, g2, correction=True)
            t_val = ttest_res['T'].values[0]
            dof = ttest_res['dof'].values[0]
            p_val = ttest_res['p-val'].values[0]
            ci_low, ci_high = ttest_res['CI95%'].values[0]
            results.append({
                'Variable': col,
                'Mean_treated': avg1,
                'Mean_control': avg2,
                'T_value': t_val,
                'DOF': dof,
                'P_value': p_val,
                'CI_lower': ci_low,
                'CI_upper': ci_high
            })
        result_df = pd.DataFrame(results)
        print(result_df)
        print(f"T-test summary done")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    gc = GroupComparison()
    for field in ['Biology', 'Computer Science', 'Economics', 'Medicine', 'Physics', 'Mathematics']:
        gc.group(field)
        gc.run_psm(field)
        gc.two_group_ttest_to_excel(field)
    end_time = datetime.datetime.now()
    print(f"It takes {(end_time - start_time).seconds} seconds")
    print('Done GroupComparison!')
