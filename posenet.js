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

    let leftEarBottom = leftEar;
    leftEarBottom.position.y = nose.position.y;
    leftEarBottom.part = 'leftEarBottom';

    let rightEarBottom = rightEar;
    rightEarBottom.position.y = nose.position.y;
    rightEarBottom.part = 'rightEarBottom';

    let neckLeft = leftShoulder;
    neckLeft.position.x = leftEar.position.x;
    neckLeft.position.y = neckLeft.position.y - (leftShoulder.position.y - nose.position.y) / 2;
    neckLeft.part = 'neckLeft';

    let neckRight = rightShoulder;
    neckRight.position.x = rightEar.position.x;
    neckRight.position.y = neckRight.position.y - (rightShoulder.position.y - nose.position.y) / 2;
    neckRight.part = 'neckRight';

    topPoints.push(leftEarBottom);
    topPoints.push(rightEarBottom);
    topPoints.push(neckLeft);
    topPoints.push(neckRight);

    return topPoints;
  }

  loadImages() {
    this.images.forEach((image, index) => {
      jimg = jimp.read(image.url).then((jimg) => {
        this.images[i].img = jimp;
      }).catch((err) => {
        console.error(err);
      });
    });
  }

  drawImages(ctx, topPoints) {
    this.images.forEach((image) => {
      let leftEarBottom = topPoints.find((point) => point.part === 'leftEarBottom');
      let rightEarBottom = topPoints.find((point) => point.part === 'rightEarBototm');
      if (image.type.includes('stud')) {
        let imageObject = image.img.bitmap.data;
        ctx.drawImage(imageObject, leftEarBottom.position.x, leftEarBottom.position.y);
        ctx.drawImage(imageObject, rightEarBottom.position.x, rightEarBottom.position.y);
      } else if (image.type.includes('earring')) {

      } else if (image.type.includes('necklace')) {

      } else if (image.type.includes('bracelet')) {

      } else if (image.type.includes('ring')) {

      }
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
          topPoints = addTopPoints(topPoints);
          drawImages(ctx, topPoints);
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