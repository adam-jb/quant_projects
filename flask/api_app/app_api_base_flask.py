from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return jsonify({'yo': "<p>Hello, World!</p>"})
    # curl http://127.0.0.1:5000/ 
    # would get you json data


if __name__ == "__main__":
    app.run(debug=True)    