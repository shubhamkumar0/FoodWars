import logging
import numpy as np
# import plotly.express as px
# import pandas as pd
# import umap.umap_ as umap

from flask import Flask, request, render_template
from gensim.models import Word2Vec
# import dash_core_components as dcc
# import dash_html_components as html

app = Flask(__name__)
model = Word2Vec.load("./word2vec.model")
logger = logging.getLogger()


@app.route("/")
def home():
    try:
        return render_template("home.html")
    except Exception as e:
        print(e)
        logger.info(e)


@app.route("/predict", methods=["POST"])
def predict():
    vec = []
    similar_ingreds = []
    ingreds = []
    if not model:
        model2 = Word2Vec.load("./word2vec.model")
    else:
        model2 = model
    l = [x for x in request.form.values()]
    print("list: {}".format(l))
    for each in l:
        try:
            if each:
                print(each)
                vec.append(model2[each.lower()])
        except KeyError:
            continue
    print("VEC: {}".format(vec))
    if len(vec)>0:
        similar_ingreds = model.wv.similar_by_vector(np.mean(vec, axis=0), topn=10)
        ingreds = [x[0] for x in similar_ingreds]
    return render_template("suggestion.html", suggestion=ingreds)


@app.route("/favicon.ico")
def random():
    return render_template("home.html")

# @app.route("/plot")
# def create_plot():
#     if not model:
#         model2 = Word2Vec.load("./word2vec.model")
#     else:
#         model2 = model
#     X = model[model.wv.vocab]
#     print("starting umap")
#     cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
#                                   n_components=3, random_state=42).fit_transform(X)
#     print("done umap")
#     df = pd.DataFrame(cluster_embedding, columns=('x', 'y', 'z'))
#     df["class"] = list(model.wv.vocab.keys())
#     print("starting plot")
#     fig = px.scatter_3d(df, x='x', y='y', z='z',
#                         hover_name='class')
#     print("end plot")
#     div=html.Div([dcc.Graph(figure=fig)])
#     return render_template("home.html", prediction_text="Suitable Ingrdients are {}".format(div))


if __name__ == '__main__':
    app.run(debug=True)
