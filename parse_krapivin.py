import argparse
import os
from pprint import pprint
import requests
import sys
import json
import sqlite3
import string

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
    print(title)
    doi = None
    title_query = '+'.join(title.split())
    # query = f'https://dl.acm.org/action/doSearch?AllField={title_query}'
    query = f'https://dl.acm.org/action/doSearch?fillQuickSearch=false&expand=all&AllField=Title%3A%28{title_query}%29'
    r = requests.get(query)
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, features='html.parser')
        # results = soup.find_all('a')
        # for r in results:
        #     doi = r.get('href', None)
        #     if r is not None:
        #         if doi.startswith('https://doi.org/') and doi.endswith(str(art_id)):
        #             return doi
        results = soup.find_all('li',
                                {'class': 'search__item issue-item-container'})
        title_src = ''.join(x.lower() for x in title
                            if x.isalnum())
        print(title_src)
        for r in results:
            item_title = r.find('h5', {'class': 'issue-item__title'})
            title_res = ''.join(x.lower() for x in item_title.text
                                if x.isalnum())
            print(title_res)
            if title_src == title_res:
                print('MATCH!')
                a = item_title.find('a')
                doi = a.get('href', None)
                if doi is not None:
                    doi = doi.replace('/doi/', 'https://doi.org/')
                else:
                    print('NO DOI')
                print(doi)
                return doi
            print('NOT MATCH!')
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
    if not args.doi and not args.labels:
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
    doi_cnt = 0  # doi exists counter
    lbl_cnt = 0  # labels exist counter
    empty_lbl_cnt = 0  # empty labels counter (no labels at acm library)
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
            doi = None
            if not os.path.exists(doi_path):
                doi = get_doi(data['title'])
                if doi is not None:
                    with open(doi_path, 'w') as f:
                        f.write(doi)
            else:
                with open(doi_path) as f:
                    doi = f.read()
                doi = None if doi == '' else doi
                if doi is None:
                    os.remove(doi_path)
                    print(f'{doi_path} removed')
            if doi is not None:
                doi_cnt += 1
            print(f'doi: {doi}')
            pbar.desc += f' doi: {doi_cnt}'
        if args.labels:
            labels = None
            if os.path.exists(doi_path):
                with open(doi_path) as f:
                    doi = f.read()
                doi = None if doi == '' else doi
            else:
                doi = None
            if doi is not None:
                if not os.path.exists(labels_path):
                    labels = get_labels(doi)
                    # pprint(labels)
                    if labels is not None:
                        with open(labels_path, 'w') as f:
                            json.dump(labels, f, indent=2)
                else:
                    with open(labels_path) as f:
                        json_str = f.read()
                    if json_str == '':
                        labels = None
                        os.remove(labels_path)
                        print(f'{labels_path} removed')
                    else:
                        labels = json.loads(json_str)
                if labels is not None:
                    lbl_cnt += 1
                    if len(labels) == 0:
                        empty_lbl_cnt += 1
            print(f'labels: {labels}')
            pbar.desc += f' lbl: {lbl_cnt}'
            pbar.desc += f' empty_lbl: {empty_lbl_cnt}'
        if not args.doi and not args.labels:  # Read doi and labels
            if os.path.exists(doi_path):
                with open(doi_path) as f:
                    doi = f.read()
                doi = None if doi == '' else doi
            else:
                doi = None
            data['doi'] = doi
            if os.path.exists(labels_path):
                with open(labels_path) as f:
                    labels = json.load(f)
            else:
                labels = None
            data['labels'] = labels
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
    if not args.doi and not args.labels:
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
