
# same as app_api.py but code is more organised, making use of flask_restful

from flask import Flask, jsonify, request
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)


# assigning 1+ method to each class
class HelloWorld(Resource):
	def get(self):
		return {'Hello': 'Welcome to the API'}

	def post(self):
		some_json = request.get_json() 
		return {'you sent': some_json}


class multiplyIt(Resource):
	def get(self, num):
		return {'result': num*10}


# Assigning processes (ie, classes) to URLs
api.add_resource(HelloWorld, '/')
api.add_resource(multiplyIt, '/multi/<int:num>')


# querying the above with json {“aa”: “bb”} might look like:
# curl -H “Content-Type: application/json” -X POST -d ‘{“aa”: “bb”}’ http://127.0.0.1:5000/




if __name__ == '__main__':
	app.run(debug=True)    # set debug=True to automatically reload app every time you save a file change

