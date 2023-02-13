import numpy as np
import gradio as gr
import cv2
import os
import subprocess
import datetime
import pickle
import random

from os import listdir
from os.path import isdir, isfile, join

# from yolov7.extract import extract

identitiesPath = 'yolov7/media/Identities'


def video_identity(video):
    return video


def changeLayout(video):
    print("nuovo video caricato")
    videoCap = cv2.VideoCapture(video)
    frames = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)
    video_time = datetime.timedelta(seconds=seconds)
    return gr.update(maximum=video_time)


def loadIdentity(identity):
    identity = identity.replace(' ', '_')
    print("Identità dichiarata: "+identity)
    firstFiles = [identitiesPath+'/'+identity+'/'+f for f in listdir(identitiesPath+'/'+identity) if isfile(join(identitiesPath+'/'+identity, f))]
    print(firstFiles)
    return random.sample(firstFiles, 3)

def verify(video, rangeFrom, rangeTo, id):
    id = id.replace(' ', '_')
    name = str(video)
    name = name.split('\\')[-1].split('.')[-2]
    name = name[:len(name) - 8]
    text = 'yolov7/media/gallery'
    path_query = 'yolov7/media/query' + '/' + name
    #gallery = identitiesPath+'/'+id
    gallery = identitiesPath
    os.system(
        f'python yolov7/extract_new.py --weights yolov7/yolov7.pt --save-detect --conf 0.25 --fromFrameMsec {rangeFrom * 1000} --toFrameMsec {rangeTo * 1000} --classes 0 --img-size 640 --source {video} --detect-folder media --nosave --detect-folder yolov7/media --video-name {name} --query')
    output = os.system(
        f'python fast-reid/simple_demo.py --config-file ./fast-reid/configs/CMDM/mgn_R50_moco.yml  --parallel --output fast-reid/logs/demo/test --video-gallery {gallery} --video-query {path_query} --opts MODEL.WEIGHTS fast-reid/pre_models/market.pth')
    print('output', output)
    with open(text[:text.rindex("/")] + '/' + 'dictImage.pkl', 'rb') as f:
        dictImage = pickle.load(f)
    match=[]
    for t in dictImage:
        match.append(t[0].split('/')[0])
        print(match)
    for i in listdir(identitiesPath+'/'+id):
        if id in match:
            #if dictImage[i] in None:
            return '✅Subject detected!✅'
    return '⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠'

def enrollment(video, rangeFrom, rangeTo, id):
    id = id.replace(' ', '_')
    os.system(
        f'python yolov7/extract_new.py --weights yolov7/yolov7.pt --save-detect --conf 0.25 --fromFrameMsec {rangeFrom * 1000} --toFrameMsec {rangeTo * 1000} --classes 0 --img-size 640 --source {video} --nosave --detect-folder yolov7/media --video-name {id} ')
    storedFrame = [identitiesPath + '/' + id + '/' + f for f in listdir(identitiesPath + '/' + id) if
                   isfile(join(identitiesPath + '/' + id, f))]
    return ['✅'+id.replace('_',' ')+' enrolled!✅', random.sample(storedFrame, 9)]


def predict(video, text, textQuery, rangeFrom, rangeTo):

    print("ha digitato gallery path: " + text)
    print("ha digitato query path: " + textQuery)
    print("range from: " + str(rangeFrom))
    print("range to: " + str(rangeTo))
    name = str(video)
    name = name.split('\\')[-1].split('.')[-2]
    name = name[:len(name) - 8]

    os.system(
        f'python yolov7/extract_new.py --weights yolov7/yolov7.pt --save-detect --conf 0.25 --fromFrameMsec {rangeFrom * 1000} --toFrameMsec {rangeTo * 1000} --classes 0 --img-size 640 --source {video} --detect-folder media --nosave --detect-folder yolov7/media --video-name {name} --query')
    # extract(source_interface=video)
    path_query = textQuery + '/' + name
    path_image = path_query + '/' + '000544.jpg'
    # output = os.system(f'python fast-reid/simple_demo.py --config-file ./fast-reid/configs/CMDM/mgn_R50_moco.yml  --parallel --output fast-reid/logs/demo/test --video-gallery yolov7/media/gallery/video2p --query {path_image} --opts MODEL.WEIGHTS fast-reid/pre_models/market.pth')

    output = os.system(
        f'python fast-reid/simple_demo.py --config-file ./fast-reid/configs/CMDM/mgn_R50_moco.yml  --parallel --output fast-reid/logs/demo/test --video-gallery yolov7/media/gallery --video-query {path_query} --opts MODEL.WEIGHTS fast-reid/pre_models/market.pth')
    print('output', output)

    with open(text[:text.rindex("/")] + '/' + 'dictImage.pkl', 'rb') as f:
        dictImage = pickle.load(f)
    print(dictImage)
    first = dictImage[0][0]
    second = dictImage[1][0]
    third = dictImage[2][0]
    fourth = dictImage[3][0]
    fifth = dictImage[4][0]
    print(first)
    output_path = [text + '/' + first, text + '/' + second, text + '/' + third, text + '/' + fourth,
                   text + '/' + fifth]
    if output == 0:
        return ["Detection completed🔍", output_path]
    else:
        return ["No detection🔍", ["https://i.kym-cdn.com/entries/icons/mobile/000/019/277/confusedtravolta.jpg"]]


with gr.Blocks() as demo:
    gr.Markdown("""# Demo Person Re-Identification""")

    ##
    ## Enrollment
    ##
    with gr.Tab(label="Enrollment"):
        gr.Markdown("""## Enrollment""")
        """ To Enroll a new Subject:
        1. Upload a Video
            - (Optional) Define the from-to range in the video
        2. Enter a Name and Surname of the subject
        3. Click Submit ###
        """
        with gr.Row():
            outputEnroll = gr.Label(num_top_classes=3)
        with gr.Row():
            with gr.Column():
                inpVideoEnroll = gr.Video()
                inpRangeFromEnroll = gr.Slider(0, 100, value=0, label="From frame")
                inpRangeToEnroll = gr.Slider(0, 100, value=100, label="To frame")
            with gr.Column():
                nameEnroll = gr.Textbox(placeholder="Name Surname", label='Insert your name*')
                galleryEnroll = gr.Gallery(
                    label="Enroll", show_label=False, elem_id="identity"
                ).style(grid=[3], height="auto")
        with gr.Row():
            submitEnroll = gr.Button("Submit")

    ##
    ## Identification
    ##
    with gr.Tab(label="Identification"):

        gr.Markdown("""## Identification """)
        """1. Upload a Video
                    - (Optional) Define the from-to range in the video
                2. Choose the process type: Query or Gallery
                3. (Optional) Define the Gallery path
                4. (Optional) Define the Query path
                5. Click Submit
                """
        with gr.Row():
            with gr.Column():
                inp_Video = gr.Video()
                inp_RangeFrom = gr.Slider(0, 100, value=0, label="From frame")
                inp_RangeTo = gr.Slider(0, 100, value=100, label="To frame")
            with gr.Column():
                output = gr.Label(num_top_classes=3)
                gallery = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery"
                ).style(grid=[3], height="auto")
        with gr.Row():
            with gr.Column():
                #inp_Radio = gr.inputs.Radio(['Query', 'Gallery'], type="value", default='Query', label='Version*')
                inp_Text = gr.Textbox(placeholder="Specify gallery path", label='Gallery Path*',
                                      value="yolov7/media/gallery")
                inp_TextQuery = gr.Textbox(placeholder="Specify query path", label='Query Path*',
                                           value="yolov7/media/query")

        with gr.Row():
            submit_button = gr.Button("Submit")

    ##
    ## Verification
    ##
    onlyDir = [f.replace('_',' ') for f in listdir(identitiesPath) if isdir(join(identitiesPath, f))]

    with gr.Tab(label="Verification"):
        gr.Markdown("""## Verification""")
        """1. Upload a Video
                            - (Optional) Define the from-to range in the video
                        2. Select your Identity
                        3. Click Submit"""
        with gr.Row():
            outputVerif = gr.Label(num_top_classes=3)
        with gr.Row():
            with gr.Column():
                inpVideoVerif = gr.Video()
                inpRangeFromVerif = gr.Slider(0, 100, value=0, label="From frame")
                inpRangeToVerif = gr.Slider(0, 100, value=100, label="To frame")
            with gr.Column():
                identitiesVerif = gr.Radio(onlyDir, type="value", label='Who are You?')
                galleryVerif = gr.Gallery(
                    label="Identity", show_label=False, elem_id="identity"
                ).style(grid=[3], height="auto")

        # with gr.Row():
        #   with gr.Column():
        #      identitiesVerif = gr.Radio(onlyDir, type="value", label='Who are You?')
        #     inp_TextQuery = gr.Textbox(placeholder="Specify query path", label='Query Path*', value="yolov7/media/query")

        with gr.Row():
            submitVerif = gr.Button("Submit")


    # Identification
    inp_Video.change(changeLayout, inp_Video, inp_RangeFrom)
    inp_Video.change(changeLayout, inp_Video, inp_RangeTo)
    submit_button.click(predict, inputs=[inp_Video, inp_Text, inp_TextQuery, inp_RangeFrom, inp_RangeTo],
                        outputs=[output, gallery])

    # Verification
    inpVideoVerif.change(changeLayout, inpVideoVerif, inpRangeFromVerif)
    inpVideoVerif.change(changeLayout, inpVideoVerif, inpRangeToVerif)
    identitiesVerif.change(loadIdentity, identitiesVerif, galleryVerif)
    submitVerif.click(verify, inputs=[inpVideoVerif, inpRangeFromVerif, inpRangeToVerif, identitiesVerif],
                      outputs=outputVerif)

    # Enrollment
    inpVideoEnroll.change(changeLayout, inpVideoEnroll, inpRangeFromEnroll)
    inpVideoEnroll.change(changeLayout, inpVideoEnroll, inpRangeToEnroll)
    submitEnroll.click(enrollment, inputs=[inpVideoEnroll, inpRangeFromEnroll, inpRangeToEnroll, nameEnroll],
                       outputs=[outputEnroll,galleryEnroll])

if __name__ == '__main__':
    demo.launch()