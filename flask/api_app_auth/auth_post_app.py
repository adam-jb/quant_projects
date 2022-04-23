

## Code to require an api_key in a POST request's json, letting you decide what to do if a valid key is or isnt given

## Might need connection to be https to prevent key being detected when POSTing


from flask import Flask, jsonify, request

app = Flask(__name__)


def is_valid(input_key):
    return input_key == '1234'



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':               # user posts info, which can be sent to model/database/whatever
        some_json = request.get_json()           # takes whatever the user sent in the API query ...

        try:
            api_key = some_json["api_key"]
        except:
            return {"message": "Please provide an api_key"}, 400


        # check if key is valid
        if request.method == "POST" and is_valid(api_key):

            # jsonify() is similar to dumps()
            return jsonify({'message':'success, your key has been accepted :)'})

        else:
            return {"message": "The provided API key is not valid"}, 403

    else:
        return jsonify({'hello': 'you sent a GET request'})


# querying the above with what should be a valid API key:
# curl -X POST http://127.0.0.1:5000/ -H 'Content-Type: application/json' -d '{"login":"my_login","api_key":"1234"}'

## invalid key
# curl -X POST http://127.0.0.1:5000/ -H 'Content-Type: application/json' -d '{"login":"my_login","api_key":"1234dd"}'

## no api_key
# curl -X POST http://127.0.0.1:5000/ -H 'Content-Type: application/json' -d '{"login":"my_login","api_ketzz_wrong_name":"1234dd"}'



if __name__ == '__main__':
    app.run(debug=True) 

