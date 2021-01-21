"""Parse documents by organizations

Pipeline:
1. Parse documents by tika
2. [Optional] Split text into paragraphs by new line or blank line
3. [Optional] Remove stopwords (NLTK)
4. [Optional] Stem (NLTK Snowball) or lemmatize (pymorphy2) words
    and apply isalnum() function
"""

import os
from tika import parser as tika_parser
from pymorphy2 import MorphAnalyzer
# from nltk.tokenize import sent_tokenize, regexp_tokenize
from nltk.tokenize import word_tokenize, line_tokenize, blankline_tokenize
import sqlite3
import argparse
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import tika
from tqdm import tqdm


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
    parser.add_argument('--path', default='\\\\trd-vm.ibrae\\СМП_НКМ\\СОИСПОЛНИТЕЛИ')
    parser.add_argument('--db', default='data/org.sqlite')
    parser.add_argument('--text_type', default='full', choices=['full',
                                                                'parablank',
                                                                'paraline'])
    parser.add_argument('--norm_type', default='raw', choices=['raw',
                                                               'stem',
                                                               'lem'])
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--init_tika', action='store_true')
    args = parser.parse_args()
    path = args.path
    text_type = args.text_type
    norm_type = args.norm_type
    db_root, db_ext = os.path.splitext(args.db)
    db_root += '_' + text_type
    db_root += '_' + norm_type
    ex_dirs = ['!Для отправки', 'Акты СИ']
    if args.stop:
        stop = stopwords.words('russian')
        db_root += '_stop'
    else:
        stop = []
    db_path = db_root + db_ext
    snow = SnowballStemmer('russian')
    morph = MorphAnalyzer()

    if args.init_tika:
        tika.initVM()
    else:
        tika.TikaClientOnly = True

    # RESET DB
    make_db(db_path)

    # LABELS
    orgs = [x for x in os.listdir(path) if x not in ex_dirs]
    print(orgs)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for i, org in enumerate(orgs):
        cur.execute(
            'INSERT INTO Labels ('
            'label_id, label_desc) VALUES '
            '({}, "{}")'.format(i, org)
        )
    conn.commit()
    conn.close()

    # FILES
    file_id = 0
    obar = tqdm(orgs)
    for i, org in enumerate(obar):
        obar.set_description(f'{org}')
        for root, dirs, files in os.walk(os.path.join(path, org)):
            fbar = tqdm(files)
            for f in fbar:
                file_path = os.path.join(root, f)
                result = tika_parser.from_file(file_path, 'http://localhost:9998/')
                # print(result)
                content, metadata = result['content'], result['metadata']
                # print(content)
                # print(file_path)
                # print(content)
                length = len(content) if content is not None else content
                fbar.set_description(f'{file_path}: {length}')
                # TYPE
                if content is None:
                    texts = ['']
                elif text_type == 'full':
                    texts = [content]
                elif text_type == 'parablank':
                    texts = []
                    for p in blankline_tokenize(content):
                        texts.append(p)
                elif text_type == 'paraline':
                    texts = []
                    for p in line_tokenize(content):
                        texts.append(p)
                else:
                    raise NotImplementedError(text_type)
                # NORM
                if norm_type == 'stem':
                    texts = [' '.join(snow.stem(x)
                                      for x in word_tokenize(y)
                                      if x.isalnum() and x.lower() not in stop)
                             for y in texts]
                elif norm_type == 'lem':
                    texts = [' '.join(morph.parse(x)[0].normal_form
                                      for x in word_tokenize(y)
                                      if x.isalnum() and x.lower() not in stop)
                             for y in texts]
                elif norm_type == 'raw':
                    if len(stop) > 0:
                        texts = [' '.join(x
                                          for x in word_tokenize(y)
                                          if x.lower() not in stop)
                                 for y in texts]
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                for t in texts:
                    cur.execute('INSERT INTO Files values (?,?,?,?)',
                                (file_id, file_path, f'{i}', t))
                    file_id += 1
                conn.commit()
                conn.close()
                # para_tokenizer = TextTilingTokenizer(w=3, k=3,
                #                                      smoothing_width=0)
                # paras = para_tokenizer.tokenize(content)
                # print(len(paras))
                # for i, p in enumerate(blankline_tokenize(content)):
                #     print(f'\nP{i+1}')
                #     print(p)
                #     for j, s in enumerate(sent_tokenize(p)):
                #         # print(f'S{j+1}: {s}')
                #         words = [x for x in word_tokenize(s)]
                #         words = [morph.parse(x)[0].normal_form for x in words]
                #         print(f'S{j+1}: {" ".join(words)}')



