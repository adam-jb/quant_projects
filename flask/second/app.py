
# using flash() Which requires get_flashed_messages() and other code in sourced base.html (aka layout.html often)

from flask import Flask, render_template, session, redirect, url_for, flash, request
from flask_script import Shell, Manager


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or \
                request.form['password'] != 'secret':
            error = 'Invalid credentials'
        else:
            flash('You were successfully logged in', 'success')  # doesnt show the colour though :(
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

"""
# To let you explore app from CLI without much joy
def make_shell_context():
    return dict(app=app)
Manager.add_command("shell", Shell(make_context=make_shell_context))
"""

if __name__ == "__main__":
    app.run()