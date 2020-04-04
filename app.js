import React from 'react';
import ReactDOM from 'react-dom';
import ReactLoading from 'react-loading';
import { markdown } from 'markdown';
const fs = require("fs");

import styles from './styles.css';

import Joints from './joints';
import PoseNet from './posenet';

/**
 * React Component for runnign neural networks and 3D graphics
 */
class App extends React.Component {

    /**
     * the class constructor
     * @param {args} props for the parent class
     */
    constructor(props) {
        super(props);
        this.state = {
            loading: true,
            webcam: true,
        }
    }

     /**
     * One of React's life cycle methods
     * Once the current React component is loaded, this function
     * initializes neural network model, graphics engine, and webcam.
     */
    async componentDidMount() {
        this.joints = new Joints();
        this.posenet = new PoseNet(this.joints, this.refs, images);
        await this.posenet.loadNetwork();
        this.setState({loading: false});
        this.posenet.startPrediction().then((webcam) => {
            this.setState({ webcam });
        });
    }

    /** Asks for webcam access if ti was denied */
    askWebCam(){
        this.posenet.startPrediction();
    }

    /**
     * React Component's render method for rendering HTML components
     */
    render() {
        return (
            <div id="container">
                <div className="row"  id="row">
                    <div className="col-12">
                        <div className=""
                            style={{display:this.state.loading ? 'none' : 'block'}}>
                            <video ref="video" id="video" playsInline/>
                            <canvas ref="output" width={480} height={640} style={{ display: this.state.webcam ? 'block' : 'none' }}/>
                            {/* <h1>Move Farther</h1> */}
                            {!this.state.webcam && <WeCamAccess/>}
                        </div>
                        <div id="loader" style={{ display: !this.state.loading ? 'none' : 'block' }}>
                            <h3 id="loadTitle">Loading ...</h3>
                            <ReactLoading type="cylon" color="grey" height={'20%'} width={'20%'} id="reactLoader"/>
                        </div>
                    </div>
                </div>
                <div ref="description" id="description"/>
            </div>
        );
    }
}

const WeCamAccess = () => (
    <div id="webcamaccess">
        <h3>The device does not have a webcam OR webcam access was denied</h3>
        <button onClick={() => window.open("https://support.google.com/chrome/answer/2693767?p=ui_voice_search&visit_id=636795900385001472-2266950072&rd=1", "_blank")}>
            Grant Webcam Access
        </button>
    </div>);

ReactDOM.render(
    <App />,
    document.getElementById('react-container')
);

