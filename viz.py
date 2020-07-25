import numpy as np
import altair as alt
import pandas as pd


def contingency_matrix(y, y_pred, df=pd.DataFrame(), X=None, X_pred=None,
                       y_labels=None, y_pred_labels=None,
                       tooltip_cols=None,
                       table_cols=None, table_widths=None,
                       width=600, height=600, href=None,
                       cmap='tableau20', filename='cm.html',
                       sort=True, sort_type='rc',
                       inter_type='mat_leg'):
    alt.data_transformers.disable_max_rows()
    tooltip_cols = [] if tooltip_cols is None else tooltip_cols
    table_cols = [] if table_cols is None else table_cols
    table_widths = [0 for _ in table_cols] if table_widths is None else table_widths
    tooltip_cols.extend(['label_id', 'label_id_pred'])
    X = X_pred if X is None else X
    df['x'], df['y'] = X[:, 0], X[:, 1]
    df['x2'], df['y2'] = X_pred[:, 0], X_pred[:, 1]
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
    # print(df[label_col])
    if inter_type == 'mat_leg':
        selection = alt.selection_multi(
            fields=[label_col, label_pred_col],
            empty='none',
            init=[{label_col: v, label_pred_col: v2}
                  for v, v2 in zip(list(df[label_col]),
                                   list(df[label_pred_col]))]
            # toggle='event.altKey'
            # toggle='event.altKey && event.shiftKey'
            )
        sel_y = alt.selection_multi(fields=[label_col],
                                    bind='legend',
                                    empty='none',
                                    toggle='event.altKey'
                                    )
        sel_y_pred = alt.selection_multi(fields=[label_pred_col],
                                         bind='legend',
                                         empty='none',
                                         toggle='event.altKey'
                                         )
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
            color=alt.condition(selection | sel_y | sel_y_pred,
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
            color=alt.condition(selection | sel_y | sel_y_pred,
                                alt.Color(label_col + ':N', sort=y_order,
                                          scale=alt.Scale(scheme=cmap),
                                          legend=alt.Legend(
                                              columns=int(np.ceil(
                                                  len(set(y))/20)),
                                              symbolLimit=300)),
                                alt.value('lightgray')),
            opacity=alt.condition(selection | sel_y | sel_y_pred,
                                  alt.value(1.0), alt.value(0.3)),
            tooltip=tooltip_cols
        ).add_selection(
            sel_y
        ).properties(
            width=width / 2,
            height=height / 2
        ).interactive()
        if href is not None:
            points = points.encode(href=href + ':N')
        points_pred = alt.Chart(df).mark_point(filled=True).encode(
            alt.X('x2:Q', axis=None),
            alt.Y('y2:Q', axis=None),
            color=alt.condition(selection | sel_y | sel_y_pred,
                                alt.Color(label_pred_col + ':N',
                                          sort=y_pred_order,
                                          scale=alt.Scale(scheme=cmap),
                                          legend=alt.Legend(
                                              columns=int(np.ceil(
                                                  len(set(y_pred))/20)),
                                              symbolLimit=300)),
                                alt.value('lightgray')),
            opacity=alt.condition(selection | sel_y | sel_y_pred,
                                  alt.value(1.0), alt.value(0.3)),
            tooltip=tooltip_cols
        ).properties(
            width=width / 2,
            height=height / 2
        ).add_selection(
            sel_y_pred
        ).interactive()
        if href is not None:
            points_pred = points_pred.encode(href=href + ':N')
        tables = []
        for c, w in zip(table_cols, table_widths):
            tables.append(alt.Chart(df).mark_text().encode(
                text=c + ':N',
                y=alt.Y('row_number:O', axis=None)
            ).transform_window(
                row_number='row_number()'
            ).transform_filter(
                selection | sel_y | sel_y_pred
            ).transform_window(
                rank='rank(row_number)'
            ).properties(title=c, width=w))
    else:
        raise ValueError(inter_type)
    table = alt.hconcat(*tables)
    cmat = ((legend + text | (points & points_pred).resolve_scale(
        color='independent')) & table)
    cmat.save(filename)
    return cmat


def contingency_matrix_2(y, y_pred, df=pd.DataFrame(), X=None, X_pred=None,
                       y_labels=None, y_pred_labels=None,
                       tooltip_cols=None,
                       table_cols=None, table_widths=None,
                       width=600, height=600, href=None,
                       cmap='tableau20', filename='cm.html',
                       sort=True, sort_type='rc',
                       inter_type='mat_leg'):
    tooltip_cols = [] if tooltip_cols is None else tooltip_cols
    table_cols = [] if table_cols is None else table_cols
    table_widths = [0 for _ in table_cols] if table_widths is None else table_widths
    tooltip_cols.extend(['label_id', 'label_id_pred'])
    X = X_pred if X is None else X
    df['x'], df['y'] = X[:, 0], X[:, 1]
    df['x2'], df['y2'] = X_pred[:, 0], X_pred[:, 1]
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
    # print(df[label_col])
    if inter_type == 'mat_leg':
        # selection = alt.selection_multi(
        #     fields=[label_col, label_pred_col],
        #     empty='none',
        #     init=[{label_col: v, label_pred_col: v2}
        #           for v, v2 in zip(list(df[label_col]),
        #                            list(df[label_pred_col]))]
        #     # toggle='event.altKey'
        #     # toggle='event.altKey && event.shiftKey'
        #     )
        # sel_y = alt.selection_multi(fields=[label_col],
        #                             bind='legend',
        #                             empty='none',
        #                             toggle='event.altKey'
        #                             )
        # sel_y_pred = alt.selection_multi(fields=[label_pred_col],
        #                                  bind='legend',
        #                                  empty='none',
        #                                  toggle='event.altKey'
        #                                  )
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
            color=alt.Color('cm:Q', legend=None),
            # color=alt.condition(selection | sel_y | sel_y_pred,
            #                     alt.Color('cm:Q', legend=None),
            #                     alt.value('lightgray')),
        ).add_selection(
            # selection
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
        # points = alt.Chart(df).mark_point(filled=True).encode(
        #     alt.X('x:Q', axis=None),
        #     alt.Y('y:Q', axis=None),
        #     color=alt.condition(selection | sel_y | sel_y_pred,
        #                         alt.Color(label_col + ':N', sort=y_order,
        #                                   scale=alt.Scale(scheme=cmap)),
        #                         alt.value('lightgray')),
        #     opacity=alt.condition(selection | sel_y | sel_y_pred,
        #                           alt.value(1.0), alt.value(0.3)),
        #     tooltip=tooltip_cols
        # ).add_selection(
        #     sel_y
        # ).properties(
        #     width=width / 2,
        #     height=height / 2
        # ).interactive()
        # if href is not None:
        #     points = points.encode(href=href + ':N')
        # points_pred = alt.Chart(df).mark_point(filled=True).encode(
        #     alt.X('x2:Q', axis=None),
        #     alt.Y('y2:Q', axis=None),
        #     color=alt.condition(selection | sel_y | sel_y_pred,
        #                         alt.Color(label_pred_col + ':N',
        #                                   sort=y_pred_order,
        #                                   scale=alt.Scale(scheme=cmap)),
        #                         alt.value('lightgray')),
        #     opacity=alt.condition(selection | sel_y | sel_y_pred,
        #                           alt.value(1.0), alt.value(0.3)),
        #     tooltip=tooltip_cols
        # ).properties(
        #     width=width / 2,
        #     height=height / 2
        # ).add_selection(
        #     sel_y_pred
        # ).interactive()
        # if href is not None:
        #     points_pred = points_pred.encode(href=href + ':N')
        # tables = []
        # for c, w in zip(table_cols, table_widths):
        #     tables.append(alt.Chart(df).mark_text().encode(
        #         text=c + ':N',
        #         y=alt.Y('row_number:O', axis=None)
        #     ).transform_window(
        #         row_number='row_number()'
        #     ).transform_filter(
        #         selection | sel_y | sel_y_pred
        #     ).transform_window(
        #         rank='rank(row_number)'
        #     ).properties(title=c, width=w))
    else:
        raise ValueError(inter_type)
    # table = alt.hconcat(*tables)
    # cmat = ((legend + text | (points & points_pred).resolve_scale(
    #     color='independent')) & table)
    cmat = legend + text
    cmat.save(filename)
    return cmat


def metrics(M, metrics_labels, types, filename='metrics_supervised.html'):
    rows = [(metrics_labels[i], types[j], types[k], m)
            for (i, j, k), m in np.ndenumerate(M)]
    columns = ['m', 't1', 't2', 'v']
    df = pd.DataFrame.from_records(rows, columns=columns)
    rating_radio = alt.binding_radio(options=metrics_labels)
    rating_select = alt.selection_single(fields=['m'],
                                         bind=rating_radio,
                                         name="Metric",
                                         empty='none',
                                         init={'m': metrics_labels[0]})
    heatmap = alt.Chart(df).encode(
        alt.X('t2:N'),
        alt.Y('t1:N'),
    ).mark_rect().encode(
        color='v:Q'
    )
    text = alt.Chart(df).encode(
        alt.X('t2:N'),
        alt.Y('t1:N'),
    ).mark_text(baseline='middle').encode(
        text=alt.Text('v:Q', format='.3f'),
        color=alt.value('black'),
        # color=alt.condition(
        #     alt.datum.v < 0.8,
        #     alt.value('black'),
        #     alt.value('white')
        # )
    )
    p = (heatmap + text).add_selection(
        rating_select
    ).transform_filter(
        rating_select
    ).properties(width=400, height=400)
    p.save(filename)
    return p


def unsupervised_metrics(M, metrics_labels, types,
                         filename='metrics_unsupervised.html'):
    rows = [(metrics_labels[i], types[j], m)
            for (i, j), m in np.ndenumerate(M)]
    columns = ['m', 't', 'v']
    df = pd.DataFrame.from_records(rows, columns=columns)
    heatmap = alt.Chart(df).encode(
        alt.X('m:N'),
        alt.Y('t:N'),
    ).mark_rect().encode(
        color='v:Q'
    )
    text = alt.Chart(df).encode(
        alt.X('m:N'),
        alt.Y('t:N'),
    ).mark_text(baseline='middle').encode(
        text=alt.Text('v:Q', format='.3f'),
        color=alt.value('black'),
        # color=alt.condition(
        #     alt.datum.v < 0.8,
        #     alt.value('black'),
        #     alt.value('white')
        # )
    )
    p = (heatmap + text).properties(width=400, height=400)
    p.save(filename)
    return p


def multi_line(source, id_vars='n_clusters',
               title='multi_line',
               filename='multi_line.html', cmap='tableau20',
               width=1000, height=500):
    # source = pd.DataFrame(np.cumsum(np.random.randn(100, 3), 0).round(2),
    #                       columns=['A', 'B', 'C'],
    #                       index=pd.RangeIndex(100, name='x'))
    # print(source.columns)
    # source = source.melt(id_vars=id_vars, var_name='category', value_name='y')
    source = source.reset_index().melt(id_vars=id_vars,
                                       var_name='score',
                                       value_name='value')

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=[id_vars], empty='all', init={id_vars: []})

    selection = alt.selection_multi(fields=['score'], bind='legend')
    # brush = alt.selection(type='interval', encodings=['x'])

    base = alt.Chart(source).encode(
        x=f'{id_vars}:N',
        y='value:Q',
        color=alt.Color('score:N',
                        scale=alt.Scale(scheme=cmap),
                        legend=alt.Legend(
                            # columns=int(np.ceil(len(set(df)) / 20)),
                            symbolLimit=300))
    )
    line = base.mark_line().encode(
        opacity=alt.condition(selection, alt.value(1), alt.value(0.05))
    ).add_selection(
       selection
    ).interactive()
    # upper = base.encode(
    #     alt.X(f'{id_vars}:N', scale=alt.Scale(domain=brush))
    # )
    # lower = base.properties(
    #     height=60
    # ).add_selection(
    #     brush
    # )
    # The basic line
    # line = alt.Chart(source).mark_line(interpolate='basis').encode(
    # line = alt.Chart(source).mark_line().encode(
    #     x=f'{id_vars}:N',
    #     y='y:Q',
    #     color=alt.Color('category:N',
    #                     # sort=y_pred_order,
    #                     scale=alt.Scale(scheme=cmap)),
    #     opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    # ).add_selection(
    #     selection
    # )
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x=f'{id_vars}:N',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = base.mark_point().encode(
        opacity=alt.condition(nearest & selection, alt.value(1), alt.value(0))
    )
    #
    # # Draw text labels near the points, and highlight based on selection
    text = base.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest & selection, 'value:Q', alt.value(' '),
                           format='$.3f')
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x=f'{id_vars}:N',
    ).transform_filter(
        nearest
    )
    #
    # # Put the five layers into a chart and bind the data
    plot = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=width, height=height, title=title
    )
    # plot = upper & lower
    # plot = base.properties(title=title)
    plot.save(filename)
    return plot
