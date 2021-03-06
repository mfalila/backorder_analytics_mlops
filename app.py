from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import json


np.set_printoptions(precision=8, suppress=False)

import prediction_service.prediction
#from prediction_service import prediction

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__,
            static_folder=static_dir,
            template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                prediction_service.prediction.validate_input(dict_req)
                req_list = [dict(request.form)]  # convert to a list
                features_df = pd.DataFrame(data=req_list)
                trans_feats = prediction_service.prediction.Predict.transform_data(features_df)
                response = prediction_service.prediction.form_response(trans_feats)
                return render_template("index.html", response=response)
            elif request.json:
                response = prediction_service.prediction.api_response(request.json)
                response = json.dumps(str(response))
                # ###response = {"response": response}
                # ###return jsonify(response)
                return response
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later."}
            error = {"error": e}
            return render_template("404.html", error=error)

    else:
        return render_template("index.html")


if __name__ == "__main__":
    # Bind to PORT if defined, otherwise default to 5000
    port = int(os.environ.get('PORT', 1234))
    app.run(host="0.0.0.0", debug=True, port=port)
