import self as self
from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle

from FIN_LENGTH_OUT_MM import df

app = Flask(__name__)


def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>FIN LENGTH OUT(MM) PREDICTION </h1>
                </br>
                </br>
                <p> Insert your CSV file, the result will automatically downloaded, please check in your download folder
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </br>
            <form action="/table" method="GET" enc-type="multipart/form-data">
            <button type="submit" class="btn btn-primary btn-block btn-large">Display Table</button>
            </form>
            </body>
        </html>
    """


@app.route('/transform', methods=["POST"])

def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    # print("file contents: ", file_contents)
    # print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))

    # load the model from disk
    loaded_model = pickle.load(open("model2.pkl", 'rb'))
    df['prediction'] = loaded_model.predict(df[['RECV_GREEN_LENGTH', 'PI_MEASLENGTH_MM', 'FIN_LENGTH_IN_MM', 'GREEN_DIAMETER']])

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename= result.csv"
    return response




# route to html page - "table"
@app.route('/table', methods=["GET"])
def table():
    # converting csv to html
    # reading the data in the csv file
    df2 = df.get(self)
    data = pd.read_csv(df2)
    return render_template('table.html', tables=[data.to_html()], titles=[''])


if __name__ == "__main__":
    app.run(debug=False, port=9001)