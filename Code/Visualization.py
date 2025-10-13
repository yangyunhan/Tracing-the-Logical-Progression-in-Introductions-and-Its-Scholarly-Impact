import seaborn as sns
import datetime
import pandas as pd
import matplotlib.pyplot as plt


class Visualization(object):
    def __init__(self):
        self.common_path = "Code&data/"
        self.output = self.common_path + 'Analysis_output/'
        self.custom_palette = {
            'IT': '#4878CF',
            'DB': '#EE854A',
            'ERQ': '#6ACC65',
            'SO': '#D65F5F',
            'SF': '#956CB4',
            'MO': '#8C613C'
        }

    def plot_summary_distribution(self, field):
        df = pd.read_csv(self.output + field + '_summary.txt', sep=";")
        df[['Writing steps', 'Position']] = df['step'].str.split('_', n=1, expand=True)
        df['Position'] = df['Position'].str.replace('_', '-', regex=False)
        df['mean'] = df['mean'].round(2)
        field_data = df[['Writing steps', 'Position', 'mean']].rename(columns={'mean': 'Value'})
        print('field_data', field_data)
        g = sns.relplot(
            data=field_data,
            x="Writing steps",
            y="Position",
            size="Value",
            hue="Writing steps",
            kind="scatter",
            sizes=(80, 3200),
            alpha=0.7,
            palette=self.custom_palette
        )
        plt.tick_params(labelsize=16)
        plt.xlabel('Logical functions', fontsize=16)
        plt.ylabel('Positions (%)', fontsize=16)
        plt.yticks(['0-17', '17-34', '34-50', '50-67', '67-84', '84-100'])
        g._legend.remove()
        plt.show()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    vi = Visualization()
    for field in ['Biology', 'Computer Science', 'Economics', 'Medicine', 'Physics', 'Mathematics']:
        vi.plot_summary_distribution(field)
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done Visualization!')
