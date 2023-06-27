import scipy
import sklearn.decomposition
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

class AbundanceTable():
    def __init__(self, tables: list):
        '''
        :param tables: a list of input abundance tables.
        '''
        if len(tables)>=1:
            self.table = pd.concat(tables, axis=1)
        else:
            self.table = tables[0]
    def pseudo_count(self,times: int = 10000):
        '''
        abundance table transformation from percentage to pseudo count.
        :param times: the pseudo total sequencing depth.
        :return:
        '''
        self.table = (self.table*times).applymap(lambda x: int(x))
        return
    def remove_all_zero_rows(self):
        '''
        remove the rows of taxon that have no abundance at all samples.
        :return:
        '''
        self.table = self.table.loc[~(self.table==0).all(axis=1), :]
        return

def LDA_training(input_abundance_table: pd.DataFrame, n_components: int, random_state: int=0):
    '''
    train the lda model
    :param input_abundance_table:  the input data
    :param n_components: how many topics the lda model should have
    :param random_state: random seed
    :return:
    '''
    lda = sklearn.decomposition.LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda.fit(input_abundance_table.T)
    topic_components = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis] # words for each topic after normalization
    topic_components = pd.DataFrame(topic_components)
    topic_components.columns = input_abundance_table.index.values
    sample_topics = lda.transform(input_abundance_table.T)
    sample_topics = pd.DataFrame(sample_topics)
    sample_topics.index = input_abundance_table.columns
    return sample_topics,topic_components, lda

def visualize_assemblages_samples(assemblages_samples: pd.DataFrame,group_list: list):
    '''
    visualize the assemblages' distribution among samples with heatmap
    :param assemblages_samples: dataframe of assamblage_sample relation, rows:samples columns:assemblages
    :param group_list: the list of grouping information
    :return:
    '''
    # generate gradient color (RGBA format)
    from colour import Color
    red = Color("red")
    colors = list(red.range_to(Color("black"), len(group_list)))
    color_array = np.concatenate(([[str(j)] * group_list[i] for i,j in enumerate(colors)]))
    # visulize the assemblages distribution among samples
    assemblage_size = np.shape(assemblages_samples)[1]
    samples_size = np.shape(assemblages_samples)[0]
    np.random.seed(1)
    fig, ax = plt.subplots()
    fig.set_size_inches(60, 10)
    dots_x = np.arange(samples_size)
    dots_y = np.arange(assemblage_size)
    x, y = np.meshgrid(dots_x, dots_y)
    data = [x.ravel(), y.ravel()]
    sc = ax.scatter(*data, c=assemblages_samples.T.values.ravel(), s=assemblages_samples.T.values.ravel() * 350, cmap='Reds')
    cb = plt.colorbar(sc)
    x_names = assemblages_samples.index.values
    x = range(len(x_names))
    plt.xticks(x, x_names, rotation=90, fontsize=10)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), color_array):  # change the xtick label color
        ticklabel.set_color(tickcolor)
    y_names = range(1, assemblage_size + 1)
    plt.yticks(range(0, assemblage_size), y_names, fontsize=20)
    plt.ylabel('Assemblage', fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    # plt.title('Assemblages distribution among samples',fontsize = 30)
    plt.grid()
    cb.set_label('Probability', fontsize=20, color="black")
    fig.savefig("Assemblages distribution among samples.svg", dpi=300)
    return

def visualize_assemblages_taxa(assemblages_taxa: pd.DataFrame):
    '''
    visulize the taxa distribution among assemblages
    :param assemblages_taxon: dataframe of assamblage_taxon relation, rows:assemblages  columns:taxa
    :return:
    '''
    # visulize the taxa distribution among assemblages
    topic_size = np.shape(assemblages_taxa)[0]
    topic_components_top20 = assemblages_taxa.loc[:, assemblages_taxa.sum().sort_values(ascending=False).index.values[
                                                     :20]]  # top 20 taxa are selected
    np.random.seed(2)
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 10)
    dots_x = np.arange(topic_size)
    dots_y = np.arange(20)
    x, y = np.meshgrid(dots_x, dots_y)
    data = [x.ravel(), y.ravel()]
    sc = ax.scatter(*data, c=topic_components_top20.T.values.ravel(), s=topic_components_top20.T.values.ravel() * 350,
                    cmap='Greens')
    cb = plt.colorbar(sc)
    y_names = topic_components_top20.columns.values
    y = range(len(y_names))
    plt.yticks(y, y_names, rotation=0, fontsize=10)
    x_names = range(1, topic_size + 1)
    plt.xticks(range(0, topic_size), x_names, fontsize=20)
    plt.xlabel('Assemblage', fontsize=20)
    plt.ylabel('Genus', fontsize=20)
    # plt.title('Taxa distribution among assemblages',fontsize = 20) # portional only top 20
    plt.grid()
    cb.set_label('Probability', fontsize=20, color="black")
    fig.savefig("Taxa distribution among assemblages.svg", dpi=300, bbox_inches='tight')
    return

def binarize(data: pd.DataFrame, threshold_percentile: int):
    '''
    binarization for sample_assamblages dataframe for running energy landscape
    :param data:
    :param threshold_percentile: the threshold of each row to binarize the abundance
    :return:
    '''
    import copy
    threshold = []
    for i in range(len(data)):
        useful_value = data.iloc[i, :][lambda x: x >0] #binarization method can be changed
        threshold.append(float(np.percentile(useful_value,[threshold_percentile],)))
    binarized_data = copy.copy(data)
    for i in range(len(data)):
        binarized_array = data.iloc[i,:].apply(lambda x: 1 if x > threshold[i] else 0)
        binarized_data.iloc[i, ] = binarized_array
    binarized_data = binarized_data.applymap(int)
    return binarized_data

def main():
    if len(sys.argv) > 2:
        abundance_data_lda_training = list()
        for i,filename in enumerate(sys.argv):
            if i > 1:
                exec('abundance_data_{} = pd.read_csv(\'{}\',index_col=0)'.format(i, filename), globals())
                exec('abundance_data_lda_training.append(abundance_data_{})'.format(i), globals())
    else:
        abundance_data_lda_training = list()
        abundance_data_lda_training.append(pd.read_csv(sys.argv[2], index_col=0))
    abundance_data_lda_to_transform = list()
    abundance_data_lda_to_transform.append(pd.read_csv(sys.argv[1], index_col=0, header=[0, 1, 2]))
    abundance_data_lda_training_table = AbundanceTable(abundance_data_lda_training)
    abundance_data_lda_training_table.pseudo_count()
    abundance_data_lda_training_table.remove_all_zero_rows()
    abundance_data_lda_training_table = abundance_data_lda_training_table.table
    lda_results = LDA_training(abundance_data_lda_training_table, 9, 0)
    lda_model = lda_results[2]
    visualize_assemblages_samples(lda_results[0], group_list=[65, 38, 27])
    visualize_assemblages_taxa(lda_results[1])
    # lda transformation for all samples
    all_samples_data = AbundanceTable(abundance_data_lda_to_transform)
    all_samples_data.pseudo_count(10000)
    all_samples_sample_assamblages = lda_model.transform(all_samples_data.table.loc[abundance_data_lda_training_table.index,].T)
    all_samples_sample_assamblages = pd.DataFrame(all_samples_sample_assamblages, index=all_samples_data.table.T.index)
    all_samples_sample_assamblages.T.index.names = ['assamblages']
    all_samples_sample_assamblages = all_samples_sample_assamblages.T
    all_samples_sample_assamblages.to_csv('all_samples_sample_assamblages.csv', mode='x')
    # binarization of the sample_assamblages dataframe for generate energy landscape input
    binarized_data_CD = binarize(all_samples_sample_assamblages['CD'].iloc[:,:260],threshold_percentile=75)
    binarized_data_UC = binarize(all_samples_sample_assamblages['UC'].iloc[:,:260], threshold_percentile=75)
    binarized_data_nonIBD = binarize(all_samples_sample_assamblages['nonIBD'].iloc[:,:260], threshold_percentile=75)
    binarized_data_CD.to_csv('binarized_data_01_CD.csv', mode='x')
    binarized_data_UC.to_csv('binarized_data_01_UC.csv', mode='x')
    binarized_data_nonIBD.to_csv('binarized_data_01_nonIBD.csv', mode='x')
    EL_input_CD = binarized_data_CD.replace(to_replace=0, value=-1)
    EL_input_UC = binarized_data_UC.replace(to_replace=0, value=-1)
    EL_input_nonIBD = binarized_data_nonIBD.replace(to_replace=0, value=-1)
    EL_input_CD.to_csv('binarized_data_CD.txt', header=False, index=False, mode='x')
    EL_input_UC.to_csv('binarized_data_UC.txt', header=False, index=False, mode='x')
    EL_input_nonIBD.to_csv('binarized_data_nonIBD.txt', header=False, index=False, mode='x')

sys.argv = ['','all_990samples_genus.csv', 'CD_genus_representive.csv','UC_genus_representive.csv','nonIBD_genus_representive.csv']
if __name__ == '__main__':
    main()