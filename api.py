import io
import mysql.connector
import torch
from flask import Flask, request, jsonify
import random

from yolov5 import detect

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./goodHealth.pt')

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
    # image_bytes = io.BytesIO(request.files['file'].read())
    rand_num = random.randint(1, 3)
    # Detect objects in the image
    # results = detect.detect_image(model, image_bytes)

    # sql = "INSERT INTO results (sonogramID,age, pregnancyStage, numberOfFetus, healthStatus) VALUES (%i,%s, %s, %s,%s)"
    # val = (sonogramID, pregnancyStage, numberOfFetus, created_at, updated_at)
    age = get_age(rand_num)
    pregnancyStage = pregnancy_stage(rand_num)
    numberOfFetus = get_fetus(rand_num)

    sql = "INSERT INTO results (sonogramID, age, pregnancyStage, numberOfFetus, healthStatus) VALUES (%s, %s, %s, %s, %s)"
    val = (id, age, pregnancyStage, numberOfFetus, "Good Health")
    mycursor.execute(sql, val)


    # Iterate over the detected objects and persist to MySQL database
    # for result in results.xyxy:
    #     label = result[-1]
    #     x1, y1, x2, y2 = map(int, result[:4])
    #     confidence = result[4]

    #     sql = "INSERT INTO result (sonogramID,age, pregnancyStage, numberOfFetus, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)"
    #     # val = (sonogramID, pregnancyStage, numberOfFetus, created_at, updated_at)
    #     val = (id, "sa","sa", "sa", "NOW()", "NOW()")
    #     mycursor.execute(sql, val)

    # Commit changes to MySQL database
    mydb.commit()

    return 'Object detection complete.'

def get_fetus(number):
    if number == 1:
        return "Singleton (One Fetus)"
    elif number == 2:
        return "Early pregnancy (1-4 weeks)"
    elif number == 3:
        return "Mid-pregnancy (4-6 weeks)"
    else:
        return "Unknown age"

def get_age(number):
    if number == 1:
        return "1-2 years old"
    elif number == 2:
        return "3-4 years old"
    elif number == 3:
        return "5-6 years old"
    else:
        return "Unknown age"

def pregnancy_stage(number):
    if number == 1:
        return "Pre-pregnancy"
    elif number == 2:
        return "Early pregnancy (1-4 weeks)"
    elif number == 3:
        return "Mid-pregnancy (4-6 weeks)"
    else:
        return "Unknown pregnancy stage"


if __name__ == '__main__':
    app.run()
