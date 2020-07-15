# from pyvis.network import Network
# import networkx as nx
# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# >>> nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# >>> nx_graph.nodes[3]['group'] = 10
# >>> nx_graph.add_node(20, size=20, title='couple', group=2)
# >>> nx_graph.add_node(21, size=15, title='couple', group=2)
# >>> nx_graph.add_edge(20, 21, weight=5)
# >>> nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
# >>> nt = Network("500px", "500px")
# # populates the nodes and edges data structures
# >>> nt.from_nx(nx_graph)
# >>> nt.show("nx.html")

from pyvis.network import Network

# net = Network()
# net.add_node(1, label="Node 1") # node id = 1 and label = Node 1
# net.add_node(2) # node id and label = 2
# net.add_nodes([1,2,3], value=[10, 100, 400], title=["I am node 1", "node 2 here", "and im node 3"], x=[21.4, 54.2, 11.2], y=[100.2, 23.54, 32.1], label=["NODE 1", "NODE 2", "NODE 3"], color=["#00ff1e", "#162347", "#dd4b39"])
#
# # >>> nodes = ["a", "b", "c", "d"]
# # >>> net.add_nodes(nodes) # node ids and labels = ["a", "b", "c", "d"]
# # >>> net.add_nodes("hello") # node ids and labels = ["h", "e", "l", "o"]
#
# # >>> ["size", "value", "title", "x", "y", "label", "color"]
#
# # net.add_node(0, label="a")
# # >>> net.add_node(1, label="b")
# net.add_edge(1, 2)
# # >>> net.add_edge(0, 1, weight=.87)
# # net.enable_physics(True)
# net.show_buttons(filter_=['physics'])
# net.show("net.html")

import argparse
import os
from pprint import pprint
import requests
import sys
import json
import sqlite3

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import cleanup, normalize


text_flags = {
    '--T': 'title',
    '--A': 'abstract',
    '--B': 'body',
    '--R': 'references',
    '--TR': 'references_pred',
    '--CTR': 'cited_by'
}


def load_text(path):
    raw_data = {v: '' for k, v in text_flags.items()}
    with open(path) as f:
        cur_flag = None
        for line in f:
            if line.strip() in text_flags:
                cur_flag = line.strip()
                continue
            else:
                raw_data[text_flags[cur_flag]] += line
    for key in ['references', 'references_pred', 'cited_by']:
        raw_data[key] = [x.strip('", ') for x in raw_data[key].split('\n') if x != '']
    return raw_data


def load_keywords(path):
    with open(path) as f:
        raw = f.read()
    return [x for x in raw.split('\n') if x != '']


def load_authors(path):
    with open(path) as f:
        raw = f.read()
    return [x for x in raw.split('\n') if x != '']


def get_doi(title):
    doi = None
    title = '+'.join(title.split())
    query = f'https://dl.acm.org/action/doSearch?AllField={title}'
    r = requests.get(query)
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, features='html.parser')
        results = soup.find_all('a')
        for r in results:
            doi = r.get('href', None)
            if r is not None:
                if doi.startswith('https://doi.org/') and doi.endswith(str(art_id)):
                    return doi
        doi = None
    else:
        print(r.status_code)
    return doi


def get_labels(doi):
    labels = None
    doi = doi.replace('https://doi.org/', 'https://dl.acm.org/doi/')
    r = requests.get(doi)
    if r.status_code == 200:
        labels = {}
        soup = BeautifulSoup(r.content, features='html.parser')
        ol = soup.find('ol', {"class": "rlist organizational-chart"})
        if ol is not None:
            for p in ol.findAll('li')[1:]:
                parent = p.find('div').text
                labels.setdefault(parent, [])
                for c in p.findAll('li', recursive=True):
                    child = c.find('div').text
                    labels.setdefault(child, []).append(parent)
    else:
        print(r.status_code)
    return labels


def make_db(path):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        'CREATE TABLE Files('
        'file_id INTEGER NOT NULL PRIMARY KEY, '
        'file_path TEXT NOT NULL, '
        'label_ids TEXT NOT NULL, '
        'text TEXT NOT NULL)')
    c.execute(
        'CREATE TABLE Labels('
        'label_id INTEGER NOT NULL PRIMARY KEY, '
        'label_desc TEXT NOT NULL)'
    )
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/all_docs_abstacts_refined')
    parser.add_argument('--doi', action='store_true')
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--db', default='data/krapivin.sqlite')
    parser.add_argument('--text_type', default='body', choices=['abstract',
                                                                'body'])
    parser.add_argument('--labels_type', default='top', choices=['top',
                                                                 'multitop',
                                                                 'multi'])
    args = parser.parse_args()
    # DB
    db_root, db_ext = os.path.splitext(args.db)
    db_root += '_' + args.text_type
    db_root += '_' + args.labels_type
    db_path = db_root + db_ext
    make_db(db_path)
    # DATA
    authors = load_authors(os.path.join(args.path, '!authors.dat'))
    # print(authors)
    authors = {int(x.split(',')[0]): [y.strip("' ") for y in x.split(',')[1:]] for x in authors[1:]}
    articles_ids = set()
    for (dirpath, dirnames, filenames) in os.walk(args.path):
        for fn in filenames:
            root, ext = os.path.splitext(fn)
            if root.isdigit():
                articles_ids.add(int(root))
    # print(articles_ids)
    # print(len(articles_ids))
    doi_cnt = 0
    lbl_cnt = 0
    pbar = tqdm(list(articles_ids))
    labels_ids = {}
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for art_id in pbar:
        text_path = os.path.join(args.path, f'{art_id}.txt')
        keywords_path = os.path.join(args.path, f'{art_id}.key')
        doi_path = os.path.join(args.path, f'{art_id}.doi')
        labels_path = os.path.join(args.path, f'{art_id}.lbl')
        # print(art_id)
        # print(authors[art_id])
        keywords = load_keywords(keywords_path)
        data = load_text(text_path)
        data['keywords'] = keywords
        # print(data['title'])
        pbar.desc = str(art_id)
        if args.doi:
            # if os.path.exists(doi_path):
            #     os.remove(doi_path)
            if not os.path.exists(doi_path):
                doi = get_doi(data['title'])
                if doi is not None:
                    with open(doi_path, 'w') as f:
                        f.write(doi)
                else:
                    with open(doi_path, 'w') as f:
                        f.write('')
            else:
                with open(doi_path) as f:
                    doi = f.read()
                doi = None if doi == '' else doi
                # print(doi)
            if doi is not None:
                doi_cnt += 1
            pbar.desc += f' doi: {doi_cnt}'
        if args.labels:
            with open(doi_path) as f:
                doi = f.read()
            doi = None if doi == '' else doi
            if doi is not None:
                if not os.path.exists(labels_path):
                    labels = get_labels(doi)
                    # pprint(labels)
                    if labels is not None:
                        with open(labels_path, 'w') as f:
                            json.dump(labels, f, indent=2)
                        lbl_cnt += 1
                    else:
                        with open(labels_path, 'w') as f:
                            f.write('')
            pbar.desc += f' lbl: {lbl_cnt}'
        # Read doi and labels
        if os.path.exists(doi_path):
            with open(doi_path) as f:
                doi = f.read()
            doi = None if doi == '' else doi
        else:
            doi = None
        data['doi'] = doi
        if os.path.exists(labels_path):
            # print(labels_path)
            with open(labels_path) as f:
                json_str = f.read()
            labels = None if json_str == '' else json.loads(json_str)
        else:
            labels = None
        data['labels'] = labels
        data['labels_ids'] = []
        if data['labels'] is not None:
            data_labels_ids = []
            if args.labels_type == 'top':
                labels_lens = {k: len(v) for k, v in labels.items()}
                if list(labels_lens.values()).count(0) == 1:
                    for k, v in labels.items():
                        if len(v) == 0:
                            labels_ids.setdefault(k, len(labels_ids))
                            data_labels_ids.append(labels_ids[k])
            elif args.labels_type == 'multitop':
                ids = []
                for k, v in labels.items():
                    if len(v) == 0:
                        labels_ids.setdefault(k, len(labels_ids))
                        data_labels_ids.append(labels_ids[k])
            elif args.labels_type == 'multi':
                for k, v in labels.items():
                    labels_ids.setdefault(k, len(labels_ids))
                    data_labels_ids.append(labels_ids[k])
                    for x in v:
                        labels_ids.setdefault(x, len(labels_ids))
                        data_labels_ids.append(labels_ids[x])
            else:
                raise ValueError(args.labels_type)
            data['labels_ids'] = data_labels_ids
            if len(data['labels_ids']) > 0:
                text = data[args.text_type]
                text = cleanup(text)
                text = normalize(text)
                c.execute('INSERT INTO Files values (?,?,?,?)',
                          (art_id,
                           data['doi'],
                           ','.join([str(x) for x in data['labels_ids']]),
                           text))
    pbar.close()
    pprint(sorted(labels_ids.items(), key=lambda x: x[1]))
    print(len(labels_ids))
    for k, v in labels_ids.items():
        c.execute(
            'INSERT INTO Labels ('
            'label_id, label_desc) VALUES '
            '({}, "{}")'.format(v, k)
        )
    conn.commit()
    conn.close()
