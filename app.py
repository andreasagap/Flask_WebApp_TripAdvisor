import numpy as np
import matplotlib.pyplot as plt
# import datetime
from datetime import datetime
from datetime import date
from flask import Flask, render_template, request, make_response, Response
from pymongo import MongoClient
import pandas as pd

import plotly
import plotly.graph_objs as go
import json
from plotly.graph_objs import Layout, Figure

from functions.mongodb import Db
app = Flask(__name__)
db = Db()


def string_to_datetime(d):
    try:
        d = datetime.strptime(d, '%B %Y').strftime('%Y-%b')
    except:
        d = datetime.strptime(d, '%b %Y').strftime('%Y-%b')
    return d

def create_plot(df):

    data = [
        go.Bar(
            x=df['date'],
            y=df['counts'],
            name='Group By Month'
        )
    ]
    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title='Plot'
    )

    fig = Figure(data=data, layout=layout)

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


@app.route('/')
def hello_world():

    data = db.find_all('acropolis')

    # convert json to DataFrame
    df = pd.DataFrame(list(data))
    print(df)
    df['date'] = df['date'].apply(lambda x: string_to_datetime(x))

    print(df.head(5))

    byMonthDf = df.groupby('date').size().reset_index(name='counts')
    byMonthDf = byMonthDf.sort_values(by='date', ascending=True)

    barChart = create_plot(byMonthDf)

    return render_template('home.html',
                           column_names=df.columns.values,
                           row_data=list(df.values.tolist()),
                           plot=barChart,
                           zip=zip,
                           **locals())



if __name__ == '__main__':
    app.run()
