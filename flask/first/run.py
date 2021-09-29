from flask import Flask, render_template, url_for, request
import pandas as pd
import json
import plotly
import plotly.express as px
import folium


app = Flask(__name__)

@app.route('/')   # to run on homepage
def make_homepage_inc_chart():

    df = pd.DataFrame({
      "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
      "Amount": [4, 1, 2, 2, 4, 5],
      "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })
    fig = px.bar(df, x="Fruit", y="Amount", color="City",    barmode="group")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) # saving as json, preserving plotly
                                    # cls calls a plotly-specific encoder from plotly library
                                    # there are other cls options (eg for numpy arrays)
            
    colours = ['Red', 'Blue', 'Black', 'Orange']

    circle_radius=50   # this is passed to D3 circle


    return render_template('grid.html', graphJSON=graphJSON, colours=colours, circle_radius=circle_radius) 



## making and viewing folium map
@app.route('/folium_map')
def folium_map():
  start_coords = (46.9540700, 142.7360300)
  folium_map = folium.Map(location=start_coords, zoom_start=14)
  #return folium_map._repr_html_()   # if you want to show the html map without navbar, etc, return this

  folium_map.save(app.root_path + '/templates/temp_map/map.html')
  return render_template('show_folium_map.html')




@app.route('/thing')  # there is nothing in thing
def make_thing():
  return render_template('thing.html') 


@app.route('/name/<name_input>')   # go to /name/Adam or /name/Erica to see
def say_hello(name_input):
  return "<h1>" + f'Hello there {name_input}' + '</h1>'




@app.route("/return_form_result" , methods=['GET', 'POST'])
def return_form_result():
    select = request.form.get('comp_select')
    return(str(select)) # just to see what selected colour is







##### for interest: these aren't actually used in the app
with app.test_request_context():
    print(url_for('make_thing'))   # shows you URL for that func

### could be useful for testing a request context
with app.test_request_context('/hello', method='POST'):
    # now you can do something with the request until the
    # end of the with block, such as basic assertions:
    assert request.path == '/hello'
    assert request.method == 'POST'








