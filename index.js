// import * as tf from '@tensorflow/tfjs-node';
const tf = require('@tensorflow/tfjs-node')

// import * as canvas from 'canvas';
const { Canvas, Image, ImageData, loadImage,createCanvas } = require('canvas')
// const { createCanvas, loadImage } = require('canvas')
// import * as faceapi from 'face-api.js';
const faceapi = require('face-api.js')
// import fetch from 'node-fetch';
const fetch = require('node-fetch')
const fs = require('fs')

// const { Canvas, Image, ImageData, loadImage } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, loadImage, fetch, createCanvas })

var cImg;
var model;
var emotion_labels = ["marah", "jijik", "takut", "bahagia", "sedih", "terkejut", "netral"];
var emotion_colors = ["#ff0000", "#00a800", "#ff4fc1", "#ffe100", "#306eff", "#ff9d00", "#7c7c7c"];
let scoreThreshold = 0.4;
// let minConfidence = 0.7;
let sizeType = '1024';
var offset_x = 10;
var offset_y = 30;
// loadModel('http://149.129.240.254/covid_19_riset/models/mobilenetv1_models/model.json')

// async function loadModel(path) {

//     model = await tf.loadLayersModel(path)
//     console.log('Berhasil load model')

// }


function preprocess(imgData) {
    return tf.tidy(() => {
        // let tensor = tf.fromPixels(imgData).toFloat();
        
        //  imgData.data = new Uint32Array(imgData.data.buffer);
        // console.log(imgData.data)
        let canvass = createCanvas(imgData.width, imgData.height);
        let ctxx = canvass.getContext('2d');
        ctxx.putImageData(imgData, imgData.width, imgData.height);
        // ctxx.drawImage(imgData, 0, 0, imgData.width, imgData.height);
        let tensor = tf.browser.fromPixels(canvass).toFloat();
        tensor = tensor.resizeBilinear([100, 100])

        tensor = tf.cast(tensor, 'float32')
        const offset = tf.scalar(255.0);
        // Normalisasi gambar
        const normalized = tensor.div(offset);
        //Penambahan dimensi agar bisa menjadi batch shape
        const batched = normalized.expandDims(0)
        return batched
    })
}


async function detect() {

    // var faceDetector = new FaceDetector();
   if(!model){
    model = await tf.loadLayersModel('http://149.129.240.254/covid_19_riset/models/mobilenetv1_models/model.json');
   }
        

    loadImage('./FotoFauzan.jpg').then(async (image) => {
        // console.log(image)
        // ctx.drawImage(image, 50, 0, 70, 70)
       
        // console.log('<img src="' + canvas.toDataURL() + '" />')
    
    var inputImgEl = image
    const {
        width,
        height
    } = faceapi.getMediaDimensions(inputImgEl)
   

    //        console.log(width, height)
    let out_canvas = createCanvas(width, height);
    // const canvas = createCanvas(200, 200)
    // const ctx = canvas.getContext('2d')
    out_canvas.width = width
    out_canvas.height = height
    let ctx = out_canvas.getContext("2d");
    let scale = 1;
    ctx.drawImage(inputImgEl,
        0, 0, inputImgEl.naturalWidth, inputImgEl.naturalHeight,
        0, 0, out_canvas.width, out_canvas.height);

    scale = out_canvas.width / inputImgEl.naturalWidth;

    console.time('detect');
    // return faceDetector.detect(inputImgEl)
    const forwardParams = {
            inputSize: parseInt(sizeType),
            scoreThreshold
        }
        console.log(inputImgEl)
        //deteksi wajah menggunakan faceapi
        const result = await faceapi.detectAllFaces(inputImgEl, new faceapi.TinyFaceDetectorOptions(forwardParams))
        console.log(result)
    if (result.length != 0) {


                            // const context = out_canvas.getContext('2d')
                            // //  context.drawImage(inputImgEl, 0, 0, width, height)

                            // let ctx = context;
                            ctx.lineWidth = 4;
                            ctx.font = "25px Arial"
                            ctx.fillText('Result', 0, 0);

                            for (var i = 0; i < result.length; i++) {
                                ctx.beginPath();
                                var item = result[i].box;
                                let s_x = Math.floor(item._x+offset_x);
                                if (item.y<offset_y){
                                    var s_y = Math.floor(item._y);
                                }
                                else{
                                    var s_y = Math.floor(item._y-offset_y);
                                }
                                let s_w = Math.floor(item._width-offset_x);
                                let s_h = Math.floor(item._height);
                                let cT = ctx.getImageData(s_x, s_y, s_w, s_h);
                                cT = preprocess(cT);
                                //prediksi ekspresi wajah menggunakan mobilenetv2 pretrained model
                                z = model.predict(cT)
                                z.print()
                                let index = z.argMax(1).dataSync()[0]
                                let label = emotion_labels[index];
                                ctx.strokeStyle = emotion_colors[index];
                                ctx.rect(s_x, s_y, s_w, s_h);
                                ctx.stroke();
                                ctx.fillStyle = emotion_colors[index];
                                ctx.fillText(label, s_x, s_y);
                                ctx.closePath();
                            }
                            const buffer = out_canvas.toBuffer('image/jpeg')
                            fs.writeFileSync('./gambar_hasil_convert.jpeg', buffer)
                            }
                        })
}

async function runModel() {
    // console.log(cImg)
    // if (cImg) {
        const Model_url = 'http://149.129.240.254/covid_19_riset/models/tiny_face_detector/tiny_face_detector_model-weights_manifest.json'
    await faceapi.loadTinyFaceDetectorModel(Model_url).catch(err => console.error(err));
        // let cT = preprocess(cImg)
        await detect()
    // } else {
    //     alert('Mohon untuk memilih gambar')
    // }
}

runModel()


