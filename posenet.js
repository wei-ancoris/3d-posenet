import * as posenet from '@tensorflow-models/posenet';
import { drawKeypoints } from './demo_util';
import jimp from 'jimp';

import Transform from './tranform';

const videoWidth = 480;
const videoHeight = 640;

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

/**
 * Posenet class for loading posenet
 * and running inferences on it
 */
export default class PoseNet {

  /**
   * the class constructor
   * @param {Joints} joints processes raw joints data from posenet
   * @param {array} _htmlelems that will be used to present results
   */
  constructor(joints, _htmlelems, images) {
    this.state = {
      algorithm: 'single-pose',
      input: {
        outputStride: 16,
        imageScaleFactor: 0.5,
      },
      singlePoseDetection: {
        minPoseConfidence: 0.1,
        minPartConfidence: 0.5,
      },
      net: null,
    };
    this.htmlElements = _htmlelems;
    this.joints = joints;
    this.transform = new Transform(this.joints);
    this.images = images;
    this.loadImages();
    this.previousPosition = {};
  }

  /** Checks whether the device is mobile or not */
  isMobile() {
    const mobile = /Android/i.test(navigator.userAgent) || /iPhone|iPad|iPod/i.test(navigator.userAgent);
    return mobile;
  }

  /** Starts webcam video */
  async loadVideo() {
    const video = await this.setupCamera();
    video.play();
    return video;
  }

  /** Sets uo webcam */
  async setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = this.htmlElements.video;
    video.width = videoWidth;
    video.height = videoHeight;

    const mobile = this.isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': {
        facingMode: 'user',
        width: mobile ? undefined : videoWidth,
        height: mobile ? undefined : videoHeight,
      },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  }

  addTopPoints(topPoints) {
    let nose = topPoints.find((point) => point.part === 'nose');
    let leftEar = topPoints.find((point) => point.part === 'leftEar');
    let rightEar = topPoints.find((point) => point.part === 'rightEar');
    let leftShoulder = topPoints.find((point) => point.part === 'leftShoulder');
    let rightShoulder = topPoints.find((point) => point.part === 'rightShoulder');

    let neckLeft = {score: leftShoulder.score, part: 'neckLeft', position: {x: leftEar.position.x, y: leftShoulder.position.y - (leftShoulder.position.y - nose.position.y) / 2}};
    let neckRight = {score: rightShoulder.score, part: 'neckRight', position: {x: rightEar.position.x, y: rightShoulder.position.y - (rightShoulder.position.y - nose.position.y) / 2}};

    topPoints.push(neckLeft);
    topPoints.push(neckRight);

    return topPoints;
  }

  loadImages() {
    this.images.forEach((image, index) => {
      jimp.read(image.url).then(async (jImg) => {
        await jImg.resize(image.realWidth, image.realHeight);
        this.images[index].jImg = jImg;
      }).catch((err) => {
        console.error(err);
      });
    });
  }

  decimal(num) {
    //num = num / 10;
    //return Math.round(num) * 10;
    return num;
  }

  drawImages(ctx, topPoints) {
    const self = this;
    this.images.forEach((image) => {
      let leftEar = topPoints.find((point) => point.part === 'leftEar' && point.score > self.state.singlePoseDetection.minPartConfidence);
      let rightEar = topPoints.find((point) => point.part === 'rightEar' && point.score > self.state.singlePoseDetection.minPartConfidence);
      let jImg = image.jImg;
      
      if (image.type.includes('stud')) {
        if (leftEar) {
          let leftPosition = {x: leftEar.position.x, y: leftEar.position.y};
          leftPosition.x -= (image.realWidth / 2) - 3;
          leftPosition.y += 10;
          self.drawJimp(ctx, jImg, leftPosition, 'leftEar');
        }
        if (rightEar) {
          let rightPosition = {x: rightEar.position.x, y: rightEar.position.y};
          rightPosition.x -= (image.realWidth / 2) - 3;
          rightPosition.y += 10;
          self.drawJimp(ctx, jImg, rightPosition, 'rightEar');
        }

      } else if (image.type.includes('earring')) {

      } else if (image.type.includes('necklace')) {

      } else if (image.type.includes('bracelet')) {

      } else if (image.type.includes('ring')) {

      }
    });
  }

  async drawJimp(ctx, jimpImage, position, positionName) {
    const self = this;
    if (! jimpImage || ! position) {
      return;
    }

    let img = new Image(jimpImage.bitmap.width, jimpImage.bitmap.height);
    img.onload = () => {
      if (positionName in self.previousPosition) {
        if (Math.abs(self.previousPosition[positionName].x - position.x) > 10 || Math.abs(self.previousPosition[positionName].y - position.y) > 12) {
          self.previousPosition[positionName] = position;
        }
      } else {
        self.previousPosition[positionName] = position;
      }
      ctx.drawImage(img, this.decimal(self.previousPosition[positionName].x), this.decimal(self.previousPosition[positionName].y));
    };
    jimpImage.getBase64(Jimp.AUTO, (err, src) => {
      img.src = src;
    });
  }

  /**
   * Detects human pse from video stream using posenet
   * @param {VideoObject} video 
   * @param {TFModel} net 
   */
  detectPoseInRealTime(video, net) {
    const canvas = this.htmlElements.output;
    const ctx = canvas.getContext('2d');
    // since images are being fed from a webcam
    const flipHorizontal = true;

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    const self = this;
    async function poseDetectionFrame() {
      // Scale an image down to a certain factor. Too large of an image will slow
      // down the GPU
      const imageScaleFactor = self.state.input.imageScaleFactor;
      const outputStride = +self.state.input.outputStride;

      let poses = [];
      let minPoseConfidence;
      let minPartConfidence;

      const pose = await self.net.estimateSinglePose(
        video, imageScaleFactor, flipHorizontal, outputStride);
      poses.push(pose);

      minPoseConfidence = +self.state.singlePoseDetection.minPoseConfidence;
      minPartConfidence = +self.state.singlePoseDetection.minPartConfidence;

      ctx.clearRect(0, 0, videoWidth, videoHeight);

      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();

      // For each pose (i.e. person) detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores
      poses.forEach(({ score, keypoints }) => {
        if (score >= minPoseConfidence) {
          self.transform.updateKeypoints(keypoints, minPartConfidence);
          const head = self.transform.head();
          let topPoints = keypoints.slice(0, 7);
          topPoints = self.addTopPoints(topPoints);
          self.drawImages(ctx, topPoints);
          const shouldMoveFarther = drawKeypoints(topPoints, minPartConfidence, ctx);

          if (shouldMoveFarther) {
            ctx.font = "30px Arial";
            ctx.fillText("Please Move Farther", Math.round(videoHeight / 2) - 100, Math.round(videoWidth / 2));
          }
          //drawSkeleton(keypoints, minPartConfidence, ctx);
        }
      });
      requestAnimationFrame(poseDetectionFrame);
    }

    poseDetectionFrame();
  }

  /** Loads the PoseNet model weights with architecture 0.75 */
  async loadNetwork() {
    this.net = await posenet.load();
  }

  /**
   * Starts predicting human pose from webcam
   */
  async startPrediction() {
    let video;
    try {
      video = await this.loadVideo();
    } catch (e) {
      return false;
    }
    this.detectPoseInRealTime(video, this.net);
    return true;
  }

}