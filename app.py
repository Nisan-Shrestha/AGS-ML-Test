from flask import Flask, render_template, request
import urllib.request

import tensorflow as tf
import tensorflow_hub as hub
import csv

# Download the model from TF Hub.
# model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
model_path = "./Model"
model = hub.load(model_path)
movenet = model.signatures['serving_default']

app = Flask(__name__)

print("\n\n\n Model downloaded \n\n\n")

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/result", methods = ['POST'])
def test():
    if 'img' not in request.files:
        return('No file part')
    file = request.files['img'].read()
    image =  file
    image = tf.compat.v1.image.decode_image(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 6, 56] tensor.
    keypoints = outputs['output_0']
    keypoints = keypoints[0][:][:].numpy()
    identified = 0
    score =[] 
    for i in range(6):
        avg_score = (keypoints[i,[5,8]].sum()/2)*100
        if (avg_score >= 35.0):
            score.append(avg_score)
            identified += 1
            print(i)
    print("Total of ", identified ," people detected")
    for i in range(identified):
        print("Person " , i," average confidence: %.2f"%score[i], "%")
    
    return render_template('result.html',identified = identified, score = score)
    
    # print(file.filename)
    return("ad")

# def hello():
#     # Load the input image.
#     image_path = 'IMG.jpg'
#     image = tf.io.read_file(image_path)

if __name__ == '__main__':
    app.run(host = "0.0.0.0")
