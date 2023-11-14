# Code taken from M. Buyle - https://github.com/aida-ugent/FIPR
import numpy as np
import pandas as pd
import networkx as nx
from os.path import join
import csv

from data_loaders import DataLoader


class PolBlogsLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "polblogs"

    def get_sens_attr_name(self):
        return "party"

    def _load(self):
        G = nx.read_gml(join(self._get_folder_path(), "polblogs.gml"))
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Only keep largest connected component.
        G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])

        blog_parties = pd.DataFrame.from_dict(G.nodes, orient='index')
        blog_parties = blog_parties.rename(columns={'value': 'party'}).drop(columns=['source'])
        blog_parties.party = blog_parties.party.map({0: 'left', 1: 'right'})

        positive_edges = np.array(list(G.edges()))
        return positive_edges, blog_parties

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }


class FacebookLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "facebook"

    def get_sens_attr_name(self):
        return "pages"

    def _load(self):

        with open(join(self._get_folder_path(), 'musae_facebook_edges.csv'), newline='') as f:
            reader_2 = csv.reader(f)
            edges_data = list(reader_2)

        del edges_data[0]

        edges_data = [(int(i[0]), int(i[1])) for i in edges_data]

        pages = pd.read_csv(join(self._get_folder_path(), 'musae_facebook_target.csv'), index_col=0)
        pages = pages.drop(columns=['page_name', 'facebook_id'])
       
        positive_edges = np.array(edges_data)

        return positive_edges, pages

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }

class CVLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "cv"

    def get_sens_attr_name(self):
        return "gender"

    def _load(self):
        edge_list = np.genfromtxt(join(self._get_folder_path(), 'cv_edges.csv'), delimiter=',',
                                  skip_header=1, dtype=int)

        classes = pd.read_csv(join(self._get_folder_path(), 'cv_target.csv'), index_col=0, delimiter=',')

        edge_list = [(i[0], i[1]) for i in edge_list]
        positive_edges = np.array(edge_list)

        return positive_edges, classes

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }

class LastFMLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "lastfm"

    def get_sens_attr_name(self):
        return "country"

    def _load(self):
        edge_list = np.genfromtxt(join(self._get_folder_path(), 'lastfm_asia_edges.csv'), delimiter=',',
                                  skip_header=1, dtype=int)

        classes = pd.read_csv(join(self._get_folder_path(), 'lastfm_asia_target.csv'), index_col=0, delimiter=',')

        edge_list = [(i[0], i[1]) for i in edge_list]
        positive_edges = np.array(edge_list)
        '''
        classes = [(i[0], i[1]) for i in classes]

        class_to_remove = [1, 2, 4, 7, 9, 12, 13]
        node_idx = []
        nodes_to_keep = []

        for i, j in classes:
            if j not in class_to_remove:
                nodes_to_keep.append(i)
                node_idx.append(i)

        # Remove nodes from classes
        classes = list(filter(lambda x: x[0] in nodes_to_keep, classes))
        classes = [i[1] for i in classes]

        # Remove nodes from edge list
        # Source nodes
        edge_list = list(filter(lambda x: x[0] in nodes_to_keep, edge_list))
        # Target nodes
        edge_list = list(filter(lambda x: x[1] in nodes_to_keep, edge_list))

        # Map new classes value
        d = {0: 0, 3: 1, 5: 2, 6: 3, 8: 4, 10: 5, 11: 6, 14: 7, 15: 8, 16: 9, 17: 10}
        classes = [d[i] for i in classes]

        # Get new nodeID
        dic = {k: v for v, k in enumerate(node_idx)}
        source, target = zip(*edge_list)

        source = [dic[i] for i in list(list(source))]
        target = [dic[i] for i in list(list(target))]

        positive_edges = list(zip(source, target))
        '''
        return positive_edges, classes

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }


class PokecLoader(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "pokec"

    def get_sens_attr_name(self):
        return "gender"

    def _load(self):
        edge_list = np.genfromtxt(join(self._get_folder_path(), 'edges_sub.csv'), delimiter=',',
                                  skip_header=1, dtype=int)

        features = pd.read_csv(join(self._get_folder_path(), 'features_sub.csv'), index_col=0,
                               delimiter=',')
        print(features.columns)
        classes = features[features['3']]
        # feat = features.drop(columns=features[3])

        edge_list = [(i[0], i[1]) for i in edge_list]
        positive_edges = np.array(edge_list)

        return positive_edges, classes

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }


class FacebookLoaderBis(DataLoader):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def get_dataset_name(self):
        return "facebook"

    def get_sens_attr_name(self):
        return "pages"

    def _load(self):
        with open(join(self._get_folder_path(), 'musae_facebook_edges.csv'), newline='') as f:
            reader_2 = csv.reader(f)
            edges_data = list(reader_2)

        del edges_data[0]

        edges_data = [(int(i[0]), int(i[1])) for i in edges_data]

        pages = pd.read_csv(join(self._get_folder_path(), 'musae_facebook_target.csv'), index_col=0)
        pages = pages.drop(columns=['page_name', 'facebook_id'])

        positive_edges = np.array(edges_data)

        return positive_edges, pages

    def _get_data_split_kwargs(self):
        return {
            'test_set_frac': 0.2,
            'directed': False
        }

