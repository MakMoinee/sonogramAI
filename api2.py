import io
import mysql.connector
import torch
from flask import Flask, request, jsonify
import random

from yolov5 import detect

app = Flask(__name__)

# Load the YOLOv5 model
healthStatus = torch.hub.load('ultralytics/yolov5', 'custom', path='./healthStatus.pt')
pregnancyStage = torch.hub.load('ultralytics/yolov5', 'custom', path='./pregnancyStage.pt')
numberOfFetus = torch.hub.load('ultralytics/yolov5', 'custom', path='./numberOfFetus.pt')

# Connect to MySQL database
mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="Develop@2021",
  database="sonogramdb"
)
mycursor = mydb.cursor()

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Check if request has a file and id field
    # if 'file' not in request.files or 'id' not in request.form:
    if 'id' not in request.form:
        return jsonify(error="Missing id field"), 400

    # Get the id from the form data
    id = request.form['id']

    # Read the image file and convert to bytes
    healthStatus_bytes = io.BytesIO(request.files['healthStatus'].read())
    pregnancyStage_bytes = io.BytesIO(request.files['pregnancyStage'].read())
    numberOfFetue_bytes = io.BytesIO(request.files['numberOfFetus'].read())
    # Detect objects in the image
    healthStatusResults = detect.detect_image(healthStatus, healthStatus_bytes)
    pregnancyStageResults = detect.detect_image(pregnancyStage, pregnancyStage_bytes)
    numberOfFetusResults = detect.detect_image(numberOfFetus, numberOfFetue_bytes)

    healthStatusStr = ""
    pregnancyStageStr = ""
    numberOfFetusStr = ""
    # Iterate over the detected objects and persist to MySQL database
    for result in healthStatusResults.xyxy:
        label = result[-1]
        x1, y1, x2, y2 = map(int, result[:4])
        confidence1 = result[4]

        if confidence1>0.5:
            healthStatusStr ="Good Health"
        else:
            healthStatusStr = ""

    for result2 in pregnancyStageResults.xyxy:
        label = result2[-1]
        x1, y1, x2, y2 = map(int, result[:4])
        confidence2 = result2[4]

        if confidence1>0.5:
            pregnancyStageStr ="Good Health"
        else:
            pregnancyStageStr = ""

    for result3 in numberOfFetusResults.xyxy:
        label = result3[-1]
        x1, y1, x2, y2 = map(int, result[:4])
        confidence3 = result3[4]

        numberOfFetusStr=confidence3

    sql = "INSERT INTO results (sonogramID, age, pregnancyStage, numberOfFetus, healthStatus) VALUES (%s, %s, %s, %s, %s)"
    val = (id, "", pregnancyStageStr, numberOfFetusStr, healthStatusStr)
    mycursor.execute(sql, val)

    # Commit changes to MySQL database
    mydb.commit()

    return 'Object detection complete.'


if __name__ == '__main__':
    app.run()
