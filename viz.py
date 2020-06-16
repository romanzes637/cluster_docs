import numpy as np
import altair as alt
import pandas as pd


def contingency_matrix(X, y, y_pred, df=pd.DataFrame(),
                       y_labels=None, y_pred_labels=None,
                       tooltip_cols=None, width=600, height=600,
                       cmap='tableau20', filename='cm.html',
                       sort=True, sort_type='rc'):
    tooltip_cols = [] if tooltip_cols is None else tooltip_cols
    tooltip_cols.extend(['label_id', 'label_id_pred'])
    df['x'], df['y'] = X[:, 0], X[:, 1]
    df['label_id'], df['label_id_pred'] = y, y_pred
    if y_labels is not None:
        df['label'] = y_labels
        tooltip_cols.append('label')
        label_col = 'label'
        y_i2l = dict(zip(y, y_labels))
    else:
        label_col = 'label_id'
    if y_pred_labels is not None:
        df['label_pred'] = y_pred_labels
        tooltip_cols.append('label_pred')
        label_pred_col = 'label_pred'
        yp_i2l = dict(zip(y_pred, y_pred_labels))
    else:
        label_pred_col = 'label_id_pred'
    if sort:  # sort y_pred
        labels_ids = sorted(list(set(y)))
        labels_pred_ids = sorted(list(set(y_pred)))
        # print(labels_ids)
        # print(labels_pred_ids)
        l2i = dict(zip(labels_ids, range(len(labels_ids))))
        lp2i = dict(zip(labels_pred_ids, range(len(labels_pred_ids))))
        cm = np.zeros((len(labels_ids), len(labels_pred_ids)))
        for label_id, label_id_pred in zip(y, y_pred):
            cm[l2i[label_id], lp2i[label_id_pred]] += 1
        print(cm)
        # y_order = np.argsort(-np.amax(cm, axis=1))
        # y_pred_order = np.argsort(-np.amax(cm, axis=0))
        if sort_type == 'rc':
            new_rows = np.lexsort((np.argmax(cm, axis=1),
                                   -np.amax(cm, axis=1)))
            new_cm_rc = cm[new_rows, :]
            new_cols = np.lexsort((np.argmax(new_cm_rc, axis=0),
                                   -np.amax(new_cm_rc, axis=0)))
            new_cm_rc = new_cm_rc[:, new_cols]
            print(new_cm_rc)
        elif sort_type == 'cr':
            new_cols = np.lexsort((np.argmax(cm, axis=0),
                                   -np.amax(cm, axis=0)))
            new_cm_cr = cm[:, new_cols]
            new_rows = np.lexsort((np.argmax(new_cm_cr, axis=1),
                                   -np.amax(new_cm_cr, axis=1)))
            new_cm_cr = new_cm_cr[new_rows, :]
            print(new_cm_cr)
        else:
            raise ValueError(sort_type)
        y_order = new_rows
        y_pred_order = new_cols
        y_order = [labels_ids[i] for i in y_order]
        y_pred_order = [labels_pred_ids[i] for i in y_pred_order]
        if y_labels is not None:
            y_order = [y_i2l[x] for x in y_order]
        if y_pred_labels is not None:
            y_pred_order = [yp_i2l[x] for x in y_pred_order]
    else:
        y_order = 'ascending'
        y_pred_order = 'ascending'
    selection = alt.selection_multi(fields=[label_col, label_pred_col])
    base = alt.Chart(df).transform_aggregate(
        cm='count()',
        groupby=[label_col, label_pred_col]
    ).encode(
        alt.X(label_pred_col + ':N', scale=alt.Scale(paddingInner=0),
              sort=y_pred_order),
        alt.Y(label_col + ':N', scale=alt.Scale(paddingInner=0),
              sort=y_order),
    ).properties(
        width=width,
        height=height
    )
    legend = base.mark_rect().encode(
        color=alt.condition(selection,
                            alt.Color('cm:Q', legend=None),
                            alt.value('lightgray')),
    ).add_selection(
        selection
    )
    text = base.mark_text(baseline='middle').encode(
        text='cm:Q',
        color=alt.value('black'),
        # color=alt.condition(
        #     alt.datum.num_cars > 100,
        #     alt.value('black'),
        #     alt.value('white')
        # )
    )
    points = alt.Chart(df).mark_point(filled=True).encode(
        alt.X('x:Q', axis=None),
        alt.Y('y:Q', axis=None),
        color=alt.condition(selection,
                            alt.Color(label_col + ':N', sort=y_order,
                                      scale=alt.Scale(scheme=cmap)),
                            alt.value('lightgray')),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.3)),
        tooltip=tooltip_cols
    ).add_selection(
        selection
    ).properties(
        width=width / 2,
        height=height / 2
    ).interactive()
    points_pred = alt.Chart(df).mark_point(filled=True).encode(
        alt.X('x:Q', axis=None),
        alt.Y('y:Q', axis=None),
        color=alt.condition(selection,
                            alt.Color(label_pred_col + ':N', sort=y_pred_order,
                                      scale=alt.Scale(scheme=cmap)),
                            alt.value('lightgray')),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.3)),
        tooltip=tooltip_cols
    ).properties(
        width=width / 2,
        height=height / 2
    ).add_selection(
        selection
    ).interactive()
    cmat = legend + text | (points & points_pred).resolve_scale(
        color='independent')
    cmat.save(filename)
    return cmat


if __name__ == '__main__':
    import sqlite3
    from vectorize import lda, cluster
    from sklearn.manifold import TSNE

    conn = sqlite3.connect('data/mouse.sqlite')
    files = pd.read_sql("SELECT * FROM Files", conn)
    labels = pd.read_sql("SELECT * FROM Labels", conn)
    print(len(files))
    print(labels)
    conn.close()
    # expand files (one label per file)
    y = [int(label_id) for i, ids in enumerate(files['label_ids'])
         for label_id in ids.split(',')]
    i2l = dict(zip(labels['label_id'], labels['label_desc']))
    y_labels = [i2l[x] for x in y]  # None
    y_pred_labels = None
    new_ids = [i for i, ids in enumerate(files['label_ids'])
               for label_id in ids.split(',')]
    new_file_id = [files['file_id'][i] for i in new_ids]
    new_file_path = [files['file_path'][i] for i in new_ids]
    new_label_ids = [files['label_ids'][i] for i in new_ids]
    ls = [','.join(i2l[int(l)] for l in x.split(',')) for x in new_label_ids]
    new_text = [files['text'][i] for i in new_ids]
    files = pd.DataFrame({
        'file_id': new_file_id,
        'file_path': new_file_path,
        'label_ids': new_label_ids,
        'labels': ls,
        'text': new_text
    })
    X = lda(files['text'])
    y_pred = cluster(X, n_clusters=len(labels))
    X_tsne = TSNE(random_state=42).fit_transform(X)
    del files['text']  # due to performance issues
    contingency_matrix(X_tsne, y, y_pred,
                       df=files,
                       tooltip_cols=['file_id',
                                     'file_path',
                                     'label_ids',
                                     'labels'],
                       y_labels=y_labels,
                       y_pred_labels=y_pred_labels,
                       cmap='tableau20',  # https://vega.github.io/vega/docs/schemes/
                       filename='cm_test.html',
                       sort=True,
                       sort_type='rc')
