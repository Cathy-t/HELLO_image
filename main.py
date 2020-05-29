import os
import uuid
import requests
from whitenoise import WhiteNoise
import time
import cv2

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

from src import sample
from src import translate

from retrieval.create_thumb_images import create_thumb_images
from retrieval.retrieval import load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query
import numpy as np
import torch

import sys
sys.path.append(r'D:\learning\计算机视觉\CV_project\caption-it-master\src\model.py')

from datetime import timedelta

UPLOAD_FOLDER = './static/images/'
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
YANDEX_API_KEY = 'YOUR API KEY HERE'
# SECRET_KEY = 'YOUR SECRET KEY FOR FLASK HERE'

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='./static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
# app.secret_key = SECRET_KEY

# retrieval
# Create thumb images.  创建缩略图 /home/sun/WorkSpace/HashCode/HashNet/pytorch/data/cifar10/test/

create_thumb_images(full_folder='./static/image_database/',
                    thumb_folder='./static/thumb_images/',
                    suffix='',
                    height=200,
                    del_former_thumb=True,
                    )
# Prepare data set.
data_loader = load_data(data_path='./static/image_database/',
                        batch_size=2,
                        shuffle=False,
                        transform='default',
                        )
# Prepare model. 加载预训练的model
model = load_model(pretrained_model='./src/model/Retrieval.pth', use_gpu=False)
print("Model load successfully!")

# Extract database features.
gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader) # torch.Size([59, 2048])
print("extract_feature successfully!")

# Picture extension supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)


# force browser to hold no cache. Otherwise old result returns.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# main directory of programme
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    basepath = os.path.dirname(__file__)  # current path
    upload_path = os.path.join(basepath, 'static/upload_image', 'query.jpg')

    try:
        # remove files created more than 5 minute ago
        os.system("find ./static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    if request.method == 'POST':

        # retrieval
        if request.form['submit'] == 'upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!',
                                       img_query='./static/upload_image/query.jpg?123456')
            else:
                f = request.files['picture']

                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html',
                                           message='Examine picture extension, png、PNG、jpg、JPG、bmp support.',
                                           img_query='./static/upload_image/query.jpg')
                else:

                    f.save(upload_path)

                    # transform image format and name with opencv.
                    img = cv2.imread(upload_path)  # 从原来的读取img
                    cv2.imwrite(os.path.join(basepath, 'static/upload_image', 'query.jpg'), img)  # 保存到 当前目录下

                    return render_template('upload_finish.html', message='Upload successfully!',
                                           img_query='./static/upload_image/query.jpg?123456')  # 点了upload之后的成功界面

        elif request.form['submit'] == 'retrieval':
            start_time = time.time()
            # Query.
            query_image = load_query_image('./static/upload_image/query.jpg')
            # Extract query features.
            query_feature = extract_feature_query(model=model, img=query_image)  # [1,2048]
            # Sort.
            similarity, index = sort_img(query_feature, gallery_feature)
            sorted_paths = [image_paths[i] for i in index]

            print(sorted_paths)  # 打印出查找之后根据相似度进行排序后的图片路径
            tmb_images = ['./static/thumb_images/' + os.path.split(sorted_path)[1] for sorted_path in sorted_paths]
            # sorted_files = [os.path.split(sorted_path)[1] for sorted_path in sorted_paths]

            return render_template('retrieval.html',
                                   message="Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time),
                                   sml1=similarity[0], sml2=similarity[1], sml3=similarity[2], sml4=similarity[3],
                                   sml5=similarity[4], sml6=similarity[5], sml7=similarity[6], sml8=similarity[7],
                                   sml9=similarity[8],
                                   img1_tmb=tmb_images[0], img2_tmb=tmb_images[1], img3_tmb=tmb_images[2],
                                   img4_tmb=tmb_images[3], img5_tmb=tmb_images[4], img6_tmb=tmb_images[5],
                                   img7_tmb=tmb_images[6], img8_tmb=tmb_images[7], img9_tmb=tmb_images[8],
                                   img_query='./static/upload_image/query.jpg?123456')
        # caption
        else:
            # check if the post request has the file part
            if 'content-file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            content_file = request.files['content-file']
            files = [content_file]
            # give unique name to each image
            content_name = str(uuid.uuid4()) + ".png"
            file_names = [content_name]
            for i, file in enumerate(files):
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
            args={
                'image' : "./static/images/" + file_names[0],
                'model_path': './src/model/CaptionModel_24.pth.tar',
                'vocab_path': './src/vocab/vocab.json',
                'embed_size': 256,
                'hidden_size': 512,
                'num_layers': 1,
            }
            # returns created caption
            caption = sample.main(args)

            try:
                ch_caption = translate.main(caption)
            except:
                ch_caption = ''

            # ch_caption = translate.main(caption)

            print(ch_caption, 567)

            params={
                'content': "./static/images/" + file_names[0],
                'caption': caption,
                'ch_caption': ch_caption,
            }
            return render_template('success.html', **params)
    return render_template('upload.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False)
