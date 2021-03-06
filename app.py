#!venv/bin/python
import os
from flask import Flask, url_for, redirect, render_template, request, abort, jsonify, send_from_directory, send_file
import wordclouds
import flask_admin
import pandas as pd
import getGeoCode
from flask_admin import helpers as admin_helpers, AdminIndexView, Admin
from flask_admin import BaseView, expose
import acropolis
import json
# Create Flask application
from matplotlib.figure import Figure

app = Flask(__name__)
app.config.from_pyfile('config.py')

def create_figure():
    fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # xs = range(100)
    # ys = [random.randint(1, 50) for x in xs]
    # axis.plot(xs, ys)
    return fig
# Flask views


@app.route('/wordcloud',methods=['GET', 'POST'])
def get_wordcloud():
    """ get your data here and return it as json """
    start = request.form.get('start')
    end = request.form.get('end')
    flag,p,n = wordclouds.apiWordCloud(start,end)
    if flag:
        return jsonify({
            "status": 200,
            "p": json.dumps(p),
            "n": json.dumps(n)
        })
    else:
        return jsonify({
            "status": -1
        })



@app.route('/acropoli_map',methods=['GET', 'POST'])
def get_acropolismap():
    return acropolis.getAcropolisMap()

@app.route('/top10city/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_file(filename,
                     mimetype='text/csv')

@app.route('/map', methods=['GET', 'POST'])
def map_static():
    country = request.form.get('country')
    return jsonify({
        "status": 200,
        "flightsArray": getGeoCode.getAirplanes(country)
    })
@app.route('/lda/<name>')
def lda(name):
    #plt.savefig('/static/images/new_plot.png')
    return render_template('admin/LDA/'+name+'_Period_LDA_Visualization.html')
@app.route('/admin')
def index():
    #plt.savefig('/static/images/new_plot.png')
    return render_template('index.html')
class MyHomeView(AdminIndexView):

    @expose('/')
    def index(self):
        return self.render('admin/index.html')


    @expose('/task2')
    def task2(self):
            return self.render('admin/task2.html')
    @expose('/task1')
    def task1(self):
        ratings_acropolis, man, woman, ages = acropolis.getAcropolisStatistics()
        len_ages = len(ages)
        return self.render('admin/task1.html', ratings_acropolis=ratings_acropolis,
                           man=man,
                           woman=woman,
                           len_ages=len_ages,
                           ages=ages)
    @expose('/task3')
    def task3(self):
            return self.render('admin/task3.html')
    @expose('/task4')
    def task4(self):
            return self.render('admin/task4.html')

    @expose('/task5')
    def task5(self):
        return self.render('admin/task5.html')
# Create admin

admin = flask_admin.Admin(
    app,
    'TripAdvisor',
    base_template='admin/my_master.html',
    template_mode='bootstrap4',
    index_view=MyHomeView(name='Home', url='/')

)
# Add model views
# #admin.add_view(CustomView(name="Custom view", endpoint='custom', menu_icon_type='fa', menu_icon_value='fa-connectdevelop',))



if __name__ == '__main__':

    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))


    # Start app
    app.run(debug=True)
    # change to "redis" and restart to cache again

    # some time later