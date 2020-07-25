import sqlite3

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization.bm25 import get_bm25_weights
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import sklearn.metrics.cluster as cluster_metrics
from sklearn.metrics.pairwise import euclidean_distances
# from hdbscan import HDBSCAN

from nltk.corpus import stopwords
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
import json
from pprint import pprint
import pandas as pd

from viz import contingency_matrix as cmat, multi_line
from viz import metrics as metrics_viz
from viz import unsupervised_metrics as unsupervised_metrics_viz


def w2v(data, model_path='data/word2vec_sg0'):
    if os.path.isfile(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(data,
                         # max_vocab_size=50000,
                         size=100, window=5, min_count=5,
                         workers=4, iter=15, sg=0)
        model.save(model_path)
        print('Model Saved:', model_path)

    result = []
    for text in data:
        word_vectors = []
        for word in text:
            if word in model.wv.vocab:
                word_vectors.append(model.wv[word])
        if len(word_vectors) > 0:
            result.append(np.mean(word_vectors, axis=0))
        else:
            result.append([0] * 100)
    return np.asarray(result)


def d2v(data, ids, model_path='data/doc2vec_dm1'):
    tagged_data = [
        TaggedDocument(words=text, tags=[text_id])
        for text, text_id in zip(data, ids)
    ]

    if os.path.isfile(model_path):
        model = Doc2Vec.load(model_path)
    else:
        model = Doc2Vec(tagged_data,
                        # max_vocab_size=50000,
                        vector_size=100, window=5, min_count=5,
                        workers=4, epochs=15, dm=1)
        model.save(model_path)
        print("Model Saved:", model_path)


    result = []
    for doc in tagged_data:
        result.append(*model[doc.tags])
    return np.asarray(result)


def lsa(corpus):
    tfidf = TfidfVectorizer(max_features=50000,
                            norm='l2').fit_transform(corpus).toarray()
    return TruncatedSVD(n_components=100).fit_transform(tfidf)


def tfidf(corpus):
    return TfidfVectorizer(max_features=50000,
                           norm='l2',
                           ngram_range=(1, 1)).fit_transform(corpus).toarray()


def bow(corpus):
    bow = CountVectorizer(max_features=50000,
                          ngram_range=(1, 1),
                          dtype=np.float).fit_transform(corpus).toarray()
    bow /= bow.sum(axis=1, keepdims=True)  # L1
    # bow /= np.linalg.norm(bow, axis=1, keepdims=True)  # L2
    bow[np.isnan(bow)] = 0
    return bow


def lda(corpus):
    tf = CountVectorizer(max_features=50000,
                         dtype=np.float).fit_transform(corpus).toarray()
    # tf /= tf.sum(axis=1, keepdims=True)  # L1
    # tf /= np.linalg.norm(tf, axis=1, keepdims=True)  # L2 BAD
    return LatentDirichletAllocation(
            n_components=100,
            random_state=42,
            n_jobs=-1,
            verbose=1).fit_transform(tf)


def bm25(data):
    vectors = np.array(get_bm25_weights(data, n_jobs=-1))
    # vectors /= vectors.sum(axis=1, keepdims=True)  # L1
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # L2
    vectors[np.isnan(vectors)] = 0
    return vectors


def test(corpus):
    np.random.seed(42)
    return np.random.rand(len(corpus), 2)


def rdf(corpus, model_path='data/dict.jsonld', verbose=0):
    with open(model_path, encoding="utf8") as f:
        data = json.load(f)
    if verbose:
        pprint(data)
    w2i, i2w, ids = {}, {}, []
    for d in data['@graph']:
        if d['@type'] != 'skos:Concept':
            continue
        i = d['@id']
        ids.append(i)
        words = []
        for key in ['skos:prefLabel', 'skos:altLabel', 'skos:hiddenLabel']:
            pairs = d.get(key, {}).items()
            for lang, labels in pairs:
                if isinstance(labels, list):
                    words.extend(labels)
                else:
                    words.append(labels)
        # print(words)
        for w in words:
            w2i.setdefault(w, []).append(len(ids) - 1)
            i2w.setdefault(len(ids) - 1, []).append(w)
    vectors = np.zeros((len(corpus), len(ids)))
    for index, row in enumerate(corpus):
        tokens = row.split()
        for k in range(1, 3):
            for j in range(0, len(tokens), k):
                t = ' '.join(tokens[j:j + k])
                i = w2i.get(t, None)
                if i is not None:
                    vectors[index, i] += 1
    vectors /= vectors.sum(axis=1, keepdims=True)  # L1
    # vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # L2
    vectors[np.isnan(vectors)] = 0
    return vectors


def topic_net(files, data_path='data/topic_net.csv'):
    df = pd.read_csv(data_path, usecols=['file_paths', 'vec'],
                     converters={'vec': lambda x: json.loads(x)})
    dim = len(df['vec'][0])
    print(f'vectors: {len(df)}, dim: {dim}')
    fp2i = {x: i for i, x in enumerate(df['file_paths'])}
    vectors = []
    for i, row in files.iterrows():
        fp = row['file_path']
        i2 = fp2i.get(fp, None)
        if i2 is not None:
            vectors.append(df['vec'][i2])
        else:
            print(fp)
            vectors.append([0 for _ in range(dim)])
    vectors = np.array(vectors)
    return vectors


def bert(vecs_path='data/bert_vectors.json'):
    with open(vecs_path) as f:
        vecs = [json.loads(x) for x in f.readlines()]
    return np.array(vecs)


def scibert(vecs_path='data/scibert_vectors.json'):
    with open(vecs_path) as f:
        vecs = np.array([json.loads(x) for x in f.readlines()])
    # vecs = TSNE(random_state=42).fit_transform(vecs)
    return vecs


def cluster(vec, n_clusters, true_labels=None):
    if n_clusters > 0:
        opt_func = 'silhouette_score'
        if opt_func == 'silhouette_score':
            opt_dir = 'max'
        elif opt_func == 'davies_bouldin_score':
            opt_dir = 'min'
        elif opt_func == 'calinski_harabasz_score':
            opt_dir = 'max'
        else:
            raise ValueError("opt_dir should be set to 'min' or 'max'")
        dx = vec.max() - vec.min()
        print(vec.min(), vec.max(), dx)
        sup_func = getattr(cluster_metrics, 'adjusted_mutual_info_score')
        epss = np.linspace(0.01*dx, dx, 1000)
        opt_func = getattr(cluster_metrics, opt_func)
        # sup_func = getattr(cluster_metrics, sup_func)
        values = {}
        # true_labels = np.array(true_labels)
        for eps in epss:
            clustering = DBSCAN(eps=eps, min_samples=2).fit(vec)
            pred_labels = clustering.labels_
            noise_mask = pred_labels != -1
            # print(noise_mask)
            n_clusters = len(set(pred_labels)) - 1
            if n_clusters > 1:
                # v = opt_func(vec[noise_mask], pred_labels[noise_mask])
                v = opt_func(vec, pred_labels)
            else:
                v = -1
            values[eps] = v
            if true_labels is not None:
                v2 = sup_func(true_labels, pred_labels)
                print(f'eps: {eps}, n_clusters: {n_clusters}, unsup_score: {v}, sup_score: {v2}')
            else:
                print(f'eps: {eps}, n_clusters: {n_clusters}, unsup_score: {v}')
        if opt_dir == 'min':
            eps = min(values.keys(), key=lambda x: values[x])
        elif opt_dir == 'max':
            eps = max(values.keys(), key=lambda x: values[x])
        else:
            raise ValueError("opt_dir should be set to 'min' or 'max'")
        print(f'eps: {eps}, unsup_score: {values[eps]}')
        clustering = DBSCAN(eps=eps, min_samples=2).fit(vec)
    # elif n_clusters == 0:
        # clustering = HDBSCAN(algorithm='best',
        #                      alpha=1.0,
        #                      approx_min_span_tree=True,
        #                      gen_min_span_tree=False,
        #                      leaf_size=40,
        #                      # memory=Memory(cachedir=None),
        #                      metric='euclidean',
        #                      min_cluster_size=5,
        #                      min_samples=None,
        #                      p=None).fit(vec)
    else:
        n_clusters, df = auto_cluster(vec, true_labels=true_labels)
        clustering = KMeans(n_clusters, random_state=42).fit(vec)
        # clustering = AgglomerativeClustering(distance_threshold=dt,
        #                                      n_clusters=None,
        #                                      # linkage='average',
        #                                      compute_full_tree=True).fit(vec)

        # clustering = KMeans(n_clusters, random_state=42).fit(vec)
        # clustering = AgglomerativeClustering(n_clusters).fit(vec)
        # clustering = GaussianMixture(n_components=n_clusters,
        #                              covariance_type='full').fit(vec)
    if hasattr(clustering, 'labels_'):
        labels = clustering.labels_
    else:
        labels = clustering.predict(vec)
    print(f'n_clusters: {len(set(labels))}')
    return labels, df


def auto_cluster(vec, opt_func='silhouette_score',
                 opt_dir=None, start=2, end=100,
                 sup_func='adjusted_mutual_info_score', true_labels=None):
    start = start if start > 1 else 2
    end = end if end > 1 else 2
    unsup_names = ['silhouette_score',
                   'davies_bouldin_score',
                   'calinski_harabasz_score']
    unsup_scores = {x: [] for x in unsup_names}
    unsup_funcs = {x: getattr(cluster_metrics, x) for x in unsup_names}
    unsup_dirs = dict(zip(unsup_names, ['max', 'min', 'max']))
    sup_names = ["adjusted_mutual_info_score", "adjusted_rand_score",
                 "fowlkes_mallows_score", "completeness_score",
                 "homogeneity_score", "v_measure_score",
                 "normalized_mutual_info_score",
                 # "mutual_info_score"
                 ]
    sup_scores = {x: [] for x in sup_names}
    sup_funcs = {x: getattr(cluster_metrics, x) for x in sup_names}
    other_scores = {x: [] for x in ['inertias',
                                    'mean_inter_distances',
                                    'median_inter_distances',
                                    'max_inter_distances',
                                    'min_inter_distances',
                                    'var_inter_distances',
                                    'mean_intra_distances',
                                    'median_intra_distances',
                                    'max_intra_distances',
                                    'min_intra_distances',
                                    'var_intra_distances']}
    size_scores = {x: [] for x in ['mean_size',
                                   'median_size',
                                   'max_size',
                                   'min_size',
                                   'var_size']}
    n_clusters = list(range(start, end + 1))
    for n in n_clusters:
        clustering = KMeans(n, random_state=42).fit(vec)
        pred_labels = clustering.labels_
        centers = clustering.cluster_centers_
        intra_distances = []
        sizes = []
        for i, center in enumerate(centers):
            vecs = vec[pred_labels == i, :]
            sizes.append(len(vecs)/len(vec))  # relative to total cluster size
            dist_matrix = euclidean_distances(vecs, center[None, :])
            intra_distances.extend(dist_matrix)
        intra_distances = np.array(intra_distances)
        other_scores['mean_intra_distances'].append(intra_distances.mean())
        other_scores['median_intra_distances'].append(np.median(intra_distances))
        other_scores['max_intra_distances'].append(intra_distances.max())
        other_scores['min_intra_distances'].append(intra_distances.min())
        other_scores['var_intra_distances'].append(intra_distances.var())
        sizes = np.array(sizes)
        size_scores['mean_size'].append(sizes.mean())
        size_scores['median_size'].append(np.median(sizes))
        size_scores['max_size'].append(sizes.max())
        size_scores['min_size'].append(sizes.min())
        size_scores['var_size'].append(sizes.var())
        for x in unsup_names:
            unsup_scores[x].append(unsup_funcs[x](vec, pred_labels))
        other_scores['inertias'].append(clustering.inertia_)
        dist_matrix = euclidean_distances(clustering.cluster_centers_)
        f_dist_matrix = dist_matrix[dist_matrix != 0]
        other_scores['mean_inter_distances'].append(f_dist_matrix.mean())
        other_scores['median_inter_distances'].append(np.median(f_dist_matrix))
        other_scores['max_inter_distances'].append(f_dist_matrix.max())
        other_scores['min_inter_distances'].append(f_dist_matrix.min())
        other_scores['var_inter_distances'].append(f_dist_matrix.var())
        # print(other_scores['mean_inter_distances'][-1], other_scores['median_inter_distances'][-1],
        #       other_scores['max_inter_distances'][-1], other_scores['min_inter_distances'][-1],
        #       other_scores['var_inter_distances'][-1])
        if true_labels is not None:
            for x in sup_names:
                sup_scores[x].append(sup_funcs[x](true_labels, pred_labels))
        print(f'n_clusters: {n}, unsup_score: {unsup_scores[opt_func][-1]},'
              f' sup_score: {sup_scores[sup_func][-1]}'
              f' inertia: {clustering.inertia_}')
    # Normalize
    for name, scores in unsup_scores.items():
        min_x, max_x = min(scores), max(scores)
        unsup_scores[name] = [(x - min_x)/(max_x - min_x) for x in scores]
    other_scores['max+min_inter_distances'] = [x + y for x, y in
                                         zip(other_scores['max_inter_distances'],
                                             other_scores['min_inter_distances'])]
    other_scores['max+min_intra_distances'] = [x + y for x, y in
                                               zip(other_scores[
                                                       'max_intra_distances'],
                                                   other_scores[
                                                       'min_intra_distances'])]
    other_scores['score'] = [x - y for x, y in zip(
        other_scores['max+min_inter_distances'],
        other_scores['max+min_intra_distances'])]
    for name, scores in other_scores.items():
        min_x, max_x = min(scores), max(scores)
        other_scores[name] = [(x - min_x) / (max_x - min_x) for x in scores]
    norm_size_scores = {f'norm_{k}': v for k, v in size_scores.items()}
    for name, scores in norm_size_scores.items():
        min_x, max_x = min(scores), max(scores)
        norm_size_scores[name] = [(x - min_x) / (max_x - min_x) for x in scores]
    norm_scores = {}
    # 0 to 1
    norm_scores['norm_max+min_inter_distances'] = [0.5*(x + y) for x, y in zip(
        other_scores['max_inter_distances'],
        other_scores['min_inter_distances'])]
    # 0 to 1
    norm_scores['norm_max+min_intra_distances'] = [0.5*(x + y) for x, y in zip(
        other_scores['max_intra_distances'],
        other_scores['min_intra_distances'])]
    # 0 to 1
    norm_scores['norm_max+min_norm_size'] = [0.5*(x + y) for x, y in zip(
        norm_size_scores['norm_max_size'],
        norm_size_scores['norm_min_size'])]
    norm_scores['norm_min-max_norm_size'] = [0.5*(y - x) for x, y in zip(
        norm_size_scores['norm_max_size'],
        norm_size_scores['norm_min_size'])]
    # -1 to 1
    norm_scores['norm_score'] = [x - y for x, y in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'])]
    norm_scores['norm_score_size'] = [x - y + z for x, y, z in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'],
        norm_scores['norm_max+min_norm_size'])]
    norm_scores['norm_score_min-max_size'] = [x - y + z for x, y, z in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'],
        norm_scores['norm_min-max_norm_size'])]
    # -1 to 2
    norm_scores['norm_score_mean_size'] = [x - y + z for x, y, z in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'],
        size_scores['mean_size'])]
    norm_scores['norm_score_median_size'] = [x - y + z for x, y, z in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'],
        size_scores['median_size'])]
    norm_scores['norm_score_norm_mean_size'] = [x - y + z for x, y, z in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'],
        norm_size_scores['norm_mean_size'])]
    norm_scores['norm_score_norm_median_size'] = [x - y + z for x, y, z in zip(
        norm_scores['norm_max+min_inter_distances'],
        norm_scores['norm_max+min_intra_distances'],
        norm_size_scores['norm_median_size'])]
    norm_scores['mean_norm_score'] = [x - y + z for x, y, z in zip(
        other_scores['mean_inter_distances'],
        other_scores['mean_intra_distances'],
        norm_size_scores['norm_mean_size'])]
    norm_scores['median_norm_score'] = [x - y + z for x, y, z in zip(
        other_scores['median_inter_distances'],
        other_scores['median_intra_distances'],
        norm_size_scores['norm_median_size'])]
    for name, scores in norm_scores.items():
        min_x, max_x = min(scores), max(scores)
        norm_scores[name] = [(x - min_x) / (max_x - min_x) for x in scores]
    data = {'n_clusters': n_clusters}
    data.update(other_scores)
    data.update(unsup_scores)
    data.update(norm_scores)
    data.update(size_scores)
    data.update(norm_size_scores)
    if true_labels is not None:
        data.update(sup_scores)
    df = pd.DataFrame.from_dict(data)
    df = df.set_index('n_clusters')
    # Find optimal number of clusters
    opt_values = unsup_scores[opt_func]
    if unsup_dirs[opt_func] == 'min':
        n = n_clusters[opt_values.index(min(opt_values))]
    elif unsup_dirs[opt_func] == 'max':
        n = n_clusters[opt_values.index(max(opt_values))]
    else:
        raise ValueError("opt_dir should be set to 'min' or 'max'")
    print(f'n_clusters: {n}')
    return n, df


def cluster_tokens(corpus, labels, vectorizer='tfidf'):
    assert len(corpus) == len(labels)
    unique_labels = list(set(labels))
    # print(unique_labels)
    labels_texts = {x: '' for x in unique_labels}
    for text, label in zip(corpus, labels):
        labels_texts[label] += text + ' '
    # pprint(labels_texts)
    if vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=50000,
                                     norm='l2',
                                     ngram_range=(1, 1))
        vecs = vectorizer.fit_transform(labels_texts.values()).toarray()
        tokens = vectorizer.get_feature_names()
    elif vectorizer == 'bow':
        vectorizer = CountVectorizer(max_features=50000,
                                     ngram_range=(1, 1),
                                     dtype=np.float)
        vecs = vectorizer.fit_transform(labels_texts.values()).toarray()
        # vecs /= vecs.sum(axis=1, keepdims=True)  # L1
        tokens = vectorizer.get_feature_names()
    else:
        raise ValueError(vectorizer)
    sorted_ids = np.argsort(-vecs, axis=1)  # from max to min
    # print(np.take_along_axis(vecs, sorted_ids, axis=1))
    labels_tokens = {x: [tokens[j] for j in sorted_ids[i] if vecs[i][j] != 0]
                     for i, x in enumerate(unique_labels)}
    return labels_tokens


def lbl2color(l):
    colors = [
        "#cc4767", "#6f312a", "#d59081", "#d14530", "#d27f35",
        "#887139", "#d2b64b", "#c7df48", "#c0d296", "#5c8f37",
        "#364d26", "#70d757", "#60db9e", "#4c8f76", "#75d4d5",
        "#6a93c1", "#616ed0", "#46316c", "#8842c4", "#bc87d0"
    ]
    return colors[l % len(colors)]


def make_corpus(files):
    stops = set(stopwords.words('english'))
    texts = [list(filter(lambda w: w not in stops, text.split()))
             for text in files['text']]
    corpus = [' '.join(x) for x in texts]
    return corpus


def vectors_factory(emb_type, dataset, files, indices=None):
    corpus = make_corpus(files)
    data = [x.split() for x in corpus]
    text_ids = list(map(int, files['file_id']))
    if emb_type == 'word2vec':
        vectors = w2v(data,
                      model_path=f'data/word2vec_sg0_{dataset}')
    elif emb_type == 'doc2vec':
        vectors = d2v(data, text_ids,
                      model_path=f'data/doc2vec_dm1_{dataset}')
    elif emb_type == 'lsa':
        vectors = lsa(corpus)
    elif emb_type == 'lda':
        vectors = lda(corpus)
    elif emb_type == 'test':
        vectors = test(corpus)
    elif emb_type == 'rdf':
        vectors = rdf(files['text'], model_path=f'data/dict_{dataset}.jsonld')
    elif emb_type == 'topic_net':
        vectors = topic_net(files,
                            data_path=f'data/topic_net_{dataset}.csv')
    elif emb_type == 'bert':
        vectors = bert()
    elif emb_type == 'scibert':
        vectors = scibert(vecs_path=f'data/scibert_{dataset}_vectors.json')
        if indices is not None:
            vectors = vectors[indices,:]
    elif emb_type == 'tfidf':
        vectors = tfidf(corpus)
    elif emb_type == 'bow':
        vectors = bow(corpus)
    elif emb_type == 'bm25':
        vectors = bm25(data)
    else:
        assert False, '{} is not implemented'.format(emb_type)
    return vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['word2vec', 'doc2vec', 'lsa', 'lda',
                                           'rdf', 'test', 'topic_net', 'bert',
                                           'scibert', 'tfidf', 'bow', 'bm25'], default=None)
    parser.add_argument('--type2', choices=['word2vec', 'doc2vec', 'lsa', 'lda',
                                            'rdf', 'test', 'topic_net', 'bert',
                                            'scibert', 'tfidf', 'bow', 'bm25'],
                        default=None)
    parser.add_argument('--labels', choices=['db', 'cluster'], default=None)
    parser.add_argument('--n_tokens', type=int, default=10)
    parser.add_argument('--tokens_vectorizer',
                        choices=['tfidf', 'bow'], default='tfidf')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--cmat', action='store_true')
    parser.add_argument('--from_file', action='store_true')
    parser.add_argument('--metrics', nargs='?', default=None,
                        const='metrics.json')
    parser.add_argument('--dataset', default='mouse')
    parser.add_argument('--n_clusters', type=int, default=42)
    args = parser.parse_args()

    conn = sqlite3.connect(f'data/{args.dataset}.sqlite')

    files = pd.read_sql('SELECT * FROM Files', conn)

    limit_files_ids = None
    # min_len = 1
    # max_len = 3000
    # limit_files_ids = [i for i, t in enumerate(files['text'])
    #                    if min_len <= len(t.split()) <= max_len]
    # files = files.iloc[limit_files_ids]
    # files.reset_index(drop=True, inplace=True)
    # print(len(files))
    if args.type is not None and args.labels is not None:
        vectors_path = os.path.join('data', '_'.join(
            [args.type, args.dataset, 'vectors.npy']))
        if args.from_file and os.path.exists(vectors_path):
            print(f'loading vectors from {vectors_path}')
            vectors = np.load(vectors_path)
        else:
            vectors = vectors_factory(args.type, args.dataset, files,
                                      limit_files_ids)
            if args.from_file:
                print(f'saving vectors to {vectors_path}')
                np.save(vectors_path, vectors)
        print(vectors.shape)

        if args.labels == 'db':
            labels = [list(map(int, ids.split(','))) for ids in files['label_ids']]
        elif args.labels == 'cluster':
            emb = TSNE(random_state=42).fit_transform(vectors)
            db_labels = pd.read_sql("SELECT * FROM Labels", conn)
            i2l = dict(zip(db_labels['label_id'], db_labels['label_desc']))
            # expand files (one label per file)
            y = [int(label_id) for i, ids in enumerate(files['label_ids'])
                 for label_id in ids.split(',')]
            # y_labels = [i2l[x] for x in y]  # None
            new_ids = [i for i, ids in enumerate(files['label_ids'])
                       for label_id in ids.split(',')]
            emb = np.array([emb[i] for i in new_ids])
            print(emb.shape)
            labels, df = cluster(emb, args.n_clusters, true_labels=y)
            plt.figure()
            ax = sns.lineplot(data=df, dashes=False).set_title(f'{args.type} {args.dataset}')
            fig = ax.get_figure()
            fig.set_size_inches(15, 15)
            fig.savefig(f'clustering_{args.type}_{args.dataset}.png')
        else:
            assert False, '{} is not implemented'.format(args.labels)

        # #vectors = PCA(n_components=30).fit_transform(vectors)
        emb = TSNE(random_state=42).fit_transform(vectors)
        # print(emb.shape)

        # corpus = make_corpus(files)
        # cts = cluster_tokens(corpus, labels, args.tokens_vectorizer)
        # cts = {k: v[:args.n_tokens] for k, v in cts.items()}

        if args.save:
            with open('{}.csv'.format(args.save), 'w') as out:
                file_path = list(files['file_path'])
                out.write('file_id\tfile_path\tlabel\n')
                for i in np.argsort(labels):
                    out.write('{}\t{}\t{}\n'.format(text_ids[i], file_path[i], labels[i]))


        # for i in range(len(emb)):
        #     plt.plot(emb[i][0], emb[i][1], marker='')
        #     if args.labels == 'db':
        #         for lbl in labels[i]:
        #             plt.text(emb[i][0], emb[i][1], str(lbl), color=lbl2color(lbl), fontsize=12)
        #     elif args.labels == 'cluster':
        #         plt.text(emb[i][0], emb[i][1], str(labels[i]), color=lbl2color(labels[i]), fontsize=12)
        #
        # plt.axis('off')
        # plt.show()
        if args.cmat:
            width = 700
            height = 700
            cmap = 'tableau20'  # https://vega.github.io/vega/docs/schemes/
            cm_sort = True
            sort_type = 'rc'
            inter_type = 'mat_leg'
            if args.type2 is None:
                db_labels = pd.read_sql("SELECT * FROM Labels", conn)
                i2l = dict(zip(db_labels['label_id'], db_labels['label_desc']))
                # expand files (one label per file)
                y = [int(label_id) for i, ids in enumerate(files['label_ids'])
                     for label_id in ids.split(',')]
                y_labels = [i2l[x] for x in y]  # None
                new_ids = [i for i, ids in enumerate(files['label_ids'])
                           for label_id in ids.split(',')]
                y_pred = [labels[i] for i in new_ids]
                # y_pred = labels  # FIXME Only for --labels cluster
                y_pred_labels = None
                X = None
                X_pred = np.array([emb[i] for i in new_ids])
                print(X_pred.shape)
                new_file_id = [files['file_id'][i] for i in new_ids]
                new_file_path = [files['file_path'][i] for i in new_ids]
                new_label_ids = [files['label_ids'][i] for i in new_ids]
                new_labels = [','.join(i2l[int(l)] for l in x.split(',')) for x in
                              new_label_ids]
                new_text = [files['text'][i] for i in new_ids]
                df = pd.DataFrame({
                    'file_id': new_file_id,
                    'file_path': new_file_path,
                    'label_ids': new_label_ids,
                    'labels': new_labels,
                    'text': new_text})
                df['labels'] = [','.join(i2l[int(l)] for l in x.split(','))
                                for x in df['label_ids']]
                df['file_name'] = [os.path.basename(x) for x in df['file_path']]
                df['file_path'] = ['//' + x if not x.startswith('http') else x
                                   for x in df['file_path']]
                # df['collection_id'] = y
                corpus = make_corpus(df)
                cts = cluster_tokens(corpus, y_pred, args.tokens_vectorizer)
                df['top_tokens'] = [' '.join(cts[i][:args.n_tokens])
                                    for i in y_pred]
                del df['text']  # due to performance issues
                tooltip_cols = ['file_id', 'file_name', 'file_path',
                                'label_ids', 'labels', 'top_tokens', 'label_id']
                table_cols = ['file_id', 'file_name', 'label',
                              'label_id_pred', 'top_tokens', 'label_id']
                href = 'file_path'
                cm_filename = f'cm_{args.type}_{args.dataset}.html'
            else:
                vectors2_path = os.path.join('data', '_'.join(
                    [args.type, args.dataset, 'vectors2.npy']))
                if args.from_file and os.path.exists(vectors2_path):
                    print(f'loading vectors2 from {vectors2_path}')
                    vectors2 = np.load(vectors2_path)
                else:
                    vectors2 = vectors_factory(args.type2, args.dataset,
                                               files, limit_files_ids)
                    if args.from_file:
                        print(f'saving vectors2 to {vectors2_path}')
                        np.save(vectors2_path, vectors2)
                if args.n_clusters == -1:
                    n_clusters = auto_cluster(vectors2)
                else:
                    n_clusters = args.n_clusters
                labels2 = cluster(vectors2, n_clusters=n_clusters)
                emb2 = TSNE(random_state=42).fit_transform(vectors2)
                print(emb2.shape)
                y = labels
                y_labels = None
                y_pred = labels2
                y_pred_labels = None
                X = emb
                X_pred = emb2
                df = files
                df['file_name'] = [os.path.basename(x) for x in df['file_path']]
                df['file_path'] = ['//' + x if not x.startswith('http') else x
                                   for x in df['file_path']]
                del df['text']  # due to performance issues
                tooltip_cols = ['file_id', 'file_name', 'file_path']
                table_cols = ['file_id', 'file_name', 'label_id', 'label_id_pred']
                href = 'file_path'
                cm_filename = f'cm_{args.type}_{args.type2}_{args.dataset}.html'
            cmat(y=y,
                 y_pred=y_pred,
                 df=df,
                 X=X,
                 X_pred=X_pred,
                 tooltip_cols=tooltip_cols,
                 table_cols=table_cols,
                 href=href,
                 width=width,
                 height=height,
                 y_labels=y_labels,
                 y_pred_labels=y_pred_labels,
                 cmap=cmap,
                 filename=cm_filename,
                 sort=cm_sort,
                 sort_type=sort_type,
                 inter_type=inter_type)
    if args.metrics is not None:
        with open(args.metrics) as f:
            metrics_conf = json.load(f)
        # supervised
        types = metrics_conf['supervised']['types']
        paths = [os.path.join('data', '_'.join([x, args.dataset, 'vectors.npy']))
                 for x in types]
        for p, t in zip(paths, types):
            print(f'{t} exists at {p}? {os.path.exists(p)}')
            if not os.path.exists(p):
                if t == 'db':
                    continue
                vec = vectors_factory(t, args.dataset, files, limit_files_ids)
                print(f'saving {t} vectors to {p}')
                np.save(p, vec)
        metrics = [getattr(cluster_metrics, x)
                   for x in metrics_conf['supervised']['metrics']]
        methods_labels = {}
        db_labels = pd.read_sql("SELECT * FROM Labels", conn)
        i2l = dict(zip(db_labels['label_id'], db_labels['label_desc']))
        # expand files (one label per file)
        true_labels = [int(label_id) for i, ids in enumerate(files['label_ids'])
                        for label_id in ids.split(',')]
        new_ids = [i for i, ids in enumerate(files['label_ids'])
                   for label_id in ids.split(',')]
        for i, (p, t) in enumerate(zip(paths, types)):
            print(f'{i} clustering {p} {t}')
            if t == 'db':
                labels = true_labels
            else:
                vec = np.load(p)
                # emb = TSNE(random_state=42).fit_transform(vec)
                emb = vec
                emb = np.array([emb[i] for i in new_ids])
                labels, df = cluster(emb, args.n_clusters,
                                     true_labels=true_labels)
                # plt.figure()
                # ax = sns.lineplot(data=df,
                #                   dashes=False,
                #                   palette=sns.color_palette('tab20',
                #                                             len(df.columns))).set_title(
                #     f'{t} {args.dataset} no tsne')
                # plt.legend(
                #     # bbox_to_anchor=(0.95, 1),
                #            loc='upper right',
                #            borderaxespad=0)
                # fig = ax.get_figure()
                # fig.set_size_inches(20, 10)
                # fig.savefig(f'clustering_{t}_{args.dataset}_no_tsne.png')
                multi_line(df,
                           title=f'{t} {args.dataset} no tsne',
                           filename=f'clustering_{t}_{args.dataset}_no_tsne.html')
                # labels = cluster(vec, args.n_clusters)
            methods_labels[t] = labels
        pprint({k: len(set(v)) for k, v in methods_labels.items()})
        M = np.zeros((len(metrics), len(types), len(types)))
        for i, (p, t) in enumerate(zip(paths, types)):
            y = methods_labels[t]
            for j, (p2, t2) in enumerate(zip(paths, types)):
                print(i, j, t, t2)
                y_pred = methods_labels[t2]
                if t == 'db' and t2 != 'db':
                    y_pred = [y[i] for i in new_ids]
                for k, m in enumerate(metrics):
                    M[k, i, j] = m(y, y_pred)
        metrics_viz(M, [m.__name__ for m in metrics], types,
                    filename=f'metrics_supervised_{args.dataset}.html')
        # unsupervised
        u_types = metrics_conf['unsupervised']['types']
        u_paths = [os.path.join('data', '_'.join([x, args.dataset, 'vectors.npy']))
                 for x in u_types]
        for p, t in zip(u_paths, u_types):
            print(f'{t} exists at {p}? {os.path.exists(p)}')
            if not os.path.exists(p):
                if t == 'db':
                    continue
                vec = vectors_factory(t, args.dataset, files, limit_files_ids)
                print(f'saving {t} vectors to {p}')
                np.save(p, vec)
        metrics = [getattr(cluster_metrics, x)
                   for x in metrics_conf['unsupervised']['metrics']]
        M = np.zeros((len(metrics), len(u_types)))
        for i, m in enumerate(metrics):
            for j, (p, t) in enumerate(zip(u_paths, u_types)):
                if not os.path.exists(p):
                    continue
                vec = np.load(p)
                emb = TSNE(random_state=42).fit_transform(vec)
                y = methods_labels[t]
                M[i, j] = m(emb, y)
        unsupervised_metrics_viz(M, [m.__name__ for m in metrics], u_types,
                                 filename=f'metrics_unsupervised_{args.dataset}.html')
        # clean
        # for p, t in zip(paths, types):
        #     if os.path.exists(p):
        #         print(f'removing {t} vectors from {p}')
        #         os.remove(p)
        # clean
        # for p, t in zip(u_paths, u_types):
        #     if os.path.exists(p):
        #         print(f'removing {t} vectors from {p}')
        #         os.remove(p)