from flask import Flask,request,render_template,jsonify
from src.pipline.prediction_pipline import PredictPipline,CustomData


application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            Gender = int(request.form.get("Gender"))
            , Age = int(request.form.get("Age"))
            , Type_of_Travel = (request.form.get("Type_of_Travel"))
            , Class = (request.form.get("Class"))
            , Flight_Distance = int(request.form.get("Flight_Distance"))
            , Inflight_wifi_service = int(request.form.get("Inflight_wifi_service"))
            , Ease_of_Online_booking = int(request.form.get("Ease_of_Online_booking"))
            , Food_and_drink = int(request.form.get("Food_and_drink"))
            , Online_boarding = int(request.form.get("Online_boarding"))
            , Seat_comfort = int(request.form.get("Seat_comfort"))
            , Inflight_entertainment = int(request.form.get("Inflight_entertainment"))
            , On_board_service = int(request.form.get("On_board_service"))
            , Leg_rooms_service = int(request.form.get("Leg_rooms_service"))
            , Baggage_handling = int(request.form.get("Baggage_handling"))
            , Checkin_service = int(request.form.get("Checkin_service")) 
            , Inflight_service = int(request.form.get("Inflight_service"))
            , Cleanliness = int(request.form.get("Cleanliness"))
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)

        result = pred

        if result == 0:
            return render_template("form.html",final_result = "Passenger Neutral or Dissatisfied in This Flight:{}".format(result))

        elif result == 1:
            return render_template("form.html",final_result = "Passenger Satisfied in This Flight:{}".format(result))



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
