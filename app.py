#!venv/bin/python
import os
from flask import Flask, url_for, redirect, render_template, request, abort

import flask_admin
from flask_admin import helpers as admin_helpers, AdminIndexView
from flask_admin import BaseView, expose

# Create Flask application
app = Flask(__name__)
app.config.from_pyfile('config.py')


# Flask views
@app.route('/admin')
def index():
    return render_template('index.html',ratings_acropolis=50)
class MyHomeView(AdminIndexView):
    @expose('/')
    def index(self):
        ratings_acropolis = '500'
        return self.render('admin/index.html', ratings_acropolis=ratings_acropolis)
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
