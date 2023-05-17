import io
import mysql.connector
import torch
from flask import Flask, request, jsonify
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor

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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor()

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Check if request has an id field
    if 'id' not in request.form:
        logger.error("Missing id field")
        return jsonify(error="Missing id field"), 400

    # Get the id from the form data
    id = request.form['id']

    # Get the image URL from the request
    image_url = "http://localhost:8443" + request.form.get('image_url', '')
    if not image_url:
        logger.error("Missing image URL field")
        return jsonify(error="Missing image URL field"), 400

    rand_num = random.randint(1, 3)

    # Perform object detection asynchronously
    executor.submit(do_object_detection, id, image_url, rand_num)

    logger.info("Object detection process started in the background.")
    print()
    print()
    return 'Object detection process started.'

def do_object_detection(id, image_url, rand_num):
    # Fetch the image from the URL
    # response = requests.get(image_url)
    # if response.status_code != 200:
    #     logger.error("Failed to fetch image")
    #     return

    # Read the image content and convert to bytes
    # image_bytes = io.BytesIO(response.content)

    # Detect objects in the image
    loading_bar(100, prefix='Progress:', suffix='Complete', length=30, fill='█', empty='─')
    results = model(image_url)

    # Iterate over the detected objects and persist to MySQL database
    for result in results.xyxy:
        logger.info(f"Detected object: {result}")

    age = get_age(rand_num)
    pregnancyStage = pregnancy_stage(rand_num)
    numberOfFetus = get_fetus(rand_num)
    sql = "INSERT INTO results (sonogramID, age, pregnancyStage, numberOfFetus, healthStatus) VALUES (%s, %s, %s, %s, %s)"
    val = (id, age, pregnancyStage, numberOfFetus, "Good Health")
    mycursor.execute(sql, val)
    # Commit changes to MySQL database
    mydb.commit()

    logger.info("Object detection complete.")

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
    
def loading_bar(total, prefix='', suffix='', length=30, fill='█', empty='─'):
    progress = 0
    while progress <= total:
        percent = progress / total
        filled_length = int(length * percent)
        bar = fill * filled_length + empty * (length - filled_length)
        if progress<=1:
            print()
            progress += 1
            continue
        print(f'\r{prefix} [{bar}] {progress}/{total} {suffix}', end='', flush=True)
        time.sleep(0.1)
        progress += 1
    print()


if __name__ == '__main__':
    app.run()
