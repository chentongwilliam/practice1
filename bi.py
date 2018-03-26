import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, widgetbox, row
from bokeh.models import ColumnDataSource, Label
from bokeh.models.widgets import Slider, Select, RadioButtonGroup
from bokeh.plotting import figure
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


x = np.random.random((20,6))
kw = [1, 1, 1, 1, 1, 2 ,2 ,2 ,2 ,2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
df = pd.DataFrame(x)
df['KW'] = kw
df = df.rename(columns={0: 'Spalt1',1: 'Spalt2',2: 'Spalt3',
                        3: 'Ueberg1', 4: 'Ueberg2' , 5: 'Ueberg3', 'KW': 'KW'})


def make_plot(source, title):
    plot = figure(plot_height=400, plot_width=800, title='Test', y_range=[0, 1])
    plot.title.text = title
    plot.line('x', 'y', source=source)
    return plot


def plot_pred(pre, real, label, title):
    plot = figure(plot_height=400, plot_width=800, title='Test', y_range=[0, 1])
    plot.title.text = title + ' predict'
    plot.line('x', 'y', color='green', legend='predi',source=pre)
    plot.line('x', 'y', color='black', legend='real', source=real)
    plot.add_layout(label, "right")
    return plot


#simulate next week
def simul(df, pos, kw, model, metrics, n_folds=5):
    X_train = np.array([i for i in range(1, df[df.KW == kw].shape[0]+1)]).reshape(-1, 1)
    y = np.array(df[pos][df.KW == kw])
    tmp_dict = {}
    if (kw + 1) not in df.KW.unique():
        X_test = np.array(X_train + df[df.KW == kw].shape[0]).reshape(-1, 1)
        pre_y = model.fit(X_train, y).predict(X_test)
        real_y = pre_y
    else:
        X_test = np.array([i for i in range(1, df[df.KW == (kw + 1)].shape[0] + 1)]).reshape(-1, 1)
        real_y = np.array(df[pos][df.KW == (kw + 1)])
        score = cross_val_score(model, X_train, y, cv=n_folds)
        pre_y = model.fit(X_train, y).predict(X_test)
        tmp_dict = {'cv_score': score}
        for m in metrics:
            tmp_dict[str(m)] = m(real_y, pre_y)
    return X_test, real_y, pre_y, tmp_dict


#initial source
x_S = range(df['Spalt1'][df.KW == df.KW.max()].shape[0])
y_S = np.array(df['Spalt1'][df.KW == df.KW.max()])
source_S = ColumnDataSource(dict(x=x_S, y=y_S))

x_U = range(df['Ueberg1'][df.KW == df.KW.max()].shape[0])
y_U = np.array(df['Ueberg1'][df.KW == df.KW.max()])
source_U = ColumnDataSource(dict(x=x_U, y=y_U))

metrics = [mean_squared_error, r2_score, explained_variance_score]
model = SVR()
X_test, real_y, pre_y, tmp_dict = simul(df=df, pos='Spalt1', kw=df.KW.max(), model=model, metrics=metrics)
source_SP_pre = ColumnDataSource(dict(x=X_test, y=pre_y))
source_SP_real = ColumnDataSource(dict(x=X_test, y=real_y))


#initial widgets
options_kw = [str(i) for i in range(df.KW.min(), df.KW.max()+1)]
kw_select_S = Select(title='KW', value=str(df.KW.max()), options=options_kw)
options_S = ['Spalt1', 'Spalt2', 'Spalt3']
Pos_select_S = Select(title='Position', value='Spalt1', options=options_S)


kw_slider_U = Slider(title='KW', value=df.KW.max(), start=df.KW.min(), end=df.KW.max(), step=1)
options_U = ['Ueberg1', 'Ueberg2', 'Ueberg3']
Pos_select_U = Select(title='Position', value='Ueberg1', options=options_U)


def update_data_S(attrname, old, new):
    newkw = int(kw_select_S.value)
    newPos = Pos_select_S.value
    plot_S.title.text = newPos
    x_S = range(df[newPos][df.KW == newkw].shape[0])
    y_S = df[newPos][df.KW == newkw]
    X_test, real_y, pre_y, tmp_dict = simul(df=df, pos=newPos, kw=newkw, model=model, metrics=metrics)
    source_S.data = dict(x=x_S, y=y_S)
    source_SP_pre.data = dict(x=X_test, y=pre_y)
    source_SP_real.data = dict(x=X_test, y=real_y)


def update_data_U(attrname, old, new):
    newkw = kw_slider_U.value
    newPos = Pos_select_U.value
    plot_U.title.text = newPos
    x_U = range(df[newPos][df.KW == newkw].shape[0])
    y_U = df[newPos][df.KW == newkw]
    source_U.data = dict(x=x_U, y=y_U)


title_S = 'Spalt1'
plot_S = make_plot(source_S, title_S)
title_U = 'Ueberg1'
plot_U = make_plot(source_U, title_U)
label = Label(text='hallo')
plot_SP = plot_pred(pre=source_SP_pre, real=source_SP_real, label=label, title=title_S)

kw_select_S.on_change('value', update_data_S)
Pos_select_S.on_change('value', update_data_S)
kw_slider_U.on_change('value', update_data_U)
Pos_select_U.on_change('value', update_data_U)


Spalt_layout = column(Pos_select_S, kw_select_S, row(plot_S, plot_SP))
Ueb_layout = column(Pos_select_U, kw_slider_U, plot_U)
curdoc().add_root(column(Spalt_layout, Ueb_layout))
curdoc().title = "Plus"