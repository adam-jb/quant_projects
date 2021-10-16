from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':               # user posts info, which can be sent to model/database/whatever
		some_json = request.get_json()           # takes whatever the user sent in the API query ...
		return jsonify({'you sent': some_json}), 201   # ... and showing it back to them, with a response code of 201
	else:
		return jsonify({'hello': 'worlds'})

# querying the above with json {“aa”: “bb”} might look like:
# curl -H “Content-Type: application/json” -X POST -d ‘{“aa”: “bb”}’ http://127.0.0.1:5000/



# an endpoint that accepts numeric input
@app.route('/multi/<int:num>', methods=['GET'])
def get_multiply10(num):
	return jsonify({'result': num*10})





if __name__ == '__main__':
	app.run(debug=True)    # set debug=True to automatically reload app every time you save a file change

