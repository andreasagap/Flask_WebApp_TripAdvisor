#!venv/bin/python
import os
from flask import Flask, url_for, redirect, render_template, request, abort, jsonify
import wordclouds
import flask_admin
from flask_admin import helpers as admin_helpers, AdminIndexView
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



@app.route('/admin')
def index():
    #plt.savefig('/static/images/new_plot.png')
    return render_template('index.html')
class MyHomeView(AdminIndexView):
    @expose('/')
    def index(self):
        ratings_acropolis,man,woman,ages = acropolis.getAcropolisStatistics()
        len_ages = len(ages)
        jsonfiles = json.loads(ages.to_json(orient='index'))
        print(jsonfiles)
        return self.render('admin/index.html', ratings_acropolis=ratings_acropolis,
                           man=man,
                           woman=woman,
                           len_ages = len_ages,
                           ages=ages)
# Create admin

admin = flask_admin.Admin(
    app,
    'My Dashboard',
    base_template='admin/my_master.html',
    template_mode='bootstrap4',
    index_view=MyHomeView(name='Home',url='/')

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
