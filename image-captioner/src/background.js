import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 299;
const INCEPTION_PATH = 'http://localhost:8080/inception/model.json';
const CAPTION_PATH = 'http://localhost:8080/mymodel/model.json';
const IDX2WORD_PATH = 'http://localhost:8080/idx2word.json';
const WORD2IDX_PATH = 'http://localhost:8080/word2idx.json';

class BackgroundProcessing {

  constructor() {
    this.imageRequests = {};
    this.addListeners();
    this.loadInception();
    this.loadDictionaries();
    this.loadCaptionModel();
  }

  addListeners() {
    chrome.webRequest.onCompleted.addListener(req => {
      if (req && req.tabId > 0) {
        this.imageRequests[req.url] = this.imageRequests[req.url] || req;
        this.analyzeImage(req.url);
      }
    }, { urls: ["<all_urls>"], types: ["image", "object"] });
  }

  async loadInception() {
    console.log('Loading InceptionV3...');
    const startTime = performance.now();
    this.inception = await tf.loadLayersModel(INCEPTION_PATH);
    this.inception.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

    const totalTime = Math.floor(performance.now() - startTime);
    console.log(`InceptionV3 loaded and initialized in ${totalTime}ms...`);
  }

  async loadCaptionModel(){
    console.log('Loading Caption Model...');
    const startTime = performance.now();
    this.captionModel = await tf.loadLayersModel(CAPTION_PATH);
    const totalTime = Math.floor(performance.now() - startTime);
    console.log(`Caption Model loaded and initialized in ${totalTime}ms...`);
  }

  async loadDictionaries(){
    console.log('Loading dictionaries...');
    const startTime = performance.now();
    fetch(IDX2WORD_PATH)
    .then(response => response.json())
    .then((out) => {
    this.idx2word = out
    var start_idx = '2';
    this.idx2word[start_idx];
    })
    .catch(err => { throw err });
    fetch(WORD2IDX_PATH)
    .then(response => response.json())
    .then((out) => {
    this.word2idx = out
    var start_word = "#START#";
    this.word2idx[start_word];
    })
    .catch(err => { throw err });
    const totalTime = Math.floor(performance.now() - startTime);
    console.log(`Dictionaries loaded in ${totalTime}ms...`);
  }


  async loadImage(src) {
    return new Promise(resolve => {
      var img = document.createElement('img');
      img.crossOrigin = "anonymous";
      img.onerror = function(e) {
        resolve(null);
      };
      img.onload = function(e) {
        if ((img.height && img.height > 299) || (img.width && img.width > 299)) {
        	let w = img.width;
        	let h = img.height;
        	let diff;
        	if (w>h){
        		diff=h/299;
        		//h = 229;
        		//w=Math.round(w/diff);
        	} else {
        		diff=w/299
        		//w = 299;
        		//h=Math.round(h/diff);
        	}
        	h=h/diff;
        	w=w/diff;
          img.width = w;
          img.height = h;
          resolve(img);
        }
        resolve(null);
      };
      img.src = src;
    });
  }

	cropImage(img) {
        const size = Math.min(img.shape[0], img.shape[1]);
        const centerHeight = img.shape[0] / 2;
        const beginHeight = centerHeight - (size / 2);
        const centerWidth = img.shape[1] / 2;
        const beginWidth = centerWidth - (size / 2);
        return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
    }


  async predict(imgElement) {
    const embeddings = tf.tidy(() => {
      const img = tf.browser.fromPixels(imgElement).toFloat();
      const cropped = this.cropImage(img);
      const normalized = cropped.div(tf.scalar(225.0)).sub(tf.scalar(0.5)).mul(tf.scalar(2.0));
      const batched = normalized.expandDims(0);
      return this.inception.predict(batched);
    });
    const argmaxCaption = tf.tidy(() => {

    	let startWord = ["#START#"];
    	let index, word;
      let flattenLayer = tf.layers.flatten();
    	while(true){
    		let parCaps = [];
    		for (let j = 0; j < startWord.length; ++j) {
                parCaps.push(this.word2idx[startWord[j]]);
            }
            let flatEmbed = flattenLayer.apply(embeddings.expandDims(0));
            parCaps = tf.tensor1d(parCaps)
                        .pad([[0, 20 - startWord.length]],1)
                        .expandDims(0);
            let predictions = this.captionModel.predict([flatEmbed,parCaps]);
            predictions = predictions.reshape([predictions.shape[1]]);
            index = predictions.argMax().dataSync();
            word = this.idx2word[index];
            startWord.push(word);            
            if(word=='#END#'||startWord.length>20)
                break;
    	}
    	console.log(startWord.join(' '));
    	startWord.shift();
        startWord.pop();
        return startWord.join(' ');
    })
    return argmaxCaption;
  }

beamPredict(imgElement, beamIdx){
    const embeddings = tf.tidy(() => {
      const img = tf.browser.fromPixels(imgElement).toFloat();
      const cropped = this.cropImage(img);
      const normalized = cropped.div(tf.scalar(225.0)).sub(tf.scalar(0.5)).mul(tf.scalar(2.0));
      const batched = normalized.expandDims(0);
      return this.inception.predict(batched);
    });
    const beamCaption = tf.tidy(()=>{
    	let startWordToken = [2];
    	let startWord = [[startWordToken,0.0]];
      	let flattenLayer = tf.layers.flatten();
      	let flatEmbed = flattenLayer.apply(embeddings.expandDims(0));
    	while(startWord[0][0].length<20){
    		console.log(startWord[0][0]);
    		let temp=[];
    		for(let s of startWord){
	            let parCaps = tf.tensor1d(s[0])
	                        .pad([[0, 20 - s[0].length]],1)
	                        .expandDims(0);
	            let predictions = this.captionModel.predict([flatEmbed,parCaps]);
	            predictions = predictions.reshape([predictions.shape[1]]);
	            let predictionsarr = predictions.dataSync();
              let words = predictionsarr.map(function(a, i) { return i; }).sort(function(a, b) { return predictionsarr[a] - predictionsarr[b]; });
              let topWords = words.slice(-beamIdx);
	           for(let w of topWords){
	            	let next_cap = s[0].slice();
	            	let prob = s[1];
	            	next_cap.push(w);
	            	prob = prob + predictionsarr[w];
	            	temp.push([next_cap,prob]);
	            }

        	}
        	startWord = temp.slice();
	        startWord.sort(function(a,b){
	            return a[1]-b[1];
	        });
	        startWord = startWord.slice(-beamIdx);
    	}
    	startWord = startWord.slice(-1)[0][0];
    	let intermediateCap = [];
    	for (let j = 0; j < startWord.length; ++j) {
                intermediateCap.push(this.idx2word[startWord[j]]);
            }
      let finalCap = [];
      for(let cap of intermediateCap) {
        if (cap!="#END#") {
        	finalCap.push(cap);
        }
        else{
        	break;
        }
      }
      console.log(finalCap.join(' '));
    	return finalCap.join(' ');
    });
    return beamCaption;

}

  async analyzeImage(src) {

    if (!this.inception) {
      console.log('Model not loaded yet, delaying...');
      setTimeout(() => { this.analyzeImage(src) }, 5000);
      return;
    }

    var meta = this.imageRequests[src];
    if (meta && meta.tabId) {
      if (!meta.predictions) {
        const img = await this.loadImage(src);
        if (img) {
          meta.predictions = this.beamPredict(img,3);
          console.log(src);
        }
      }

      if (meta.predictions) {
        chrome.tabs.sendMessage(meta.tabId, {
          action: 'IMAGE_PROCESSED',
          payload: meta,
        });
      }
    }
  }
}

var bg = new BackgroundProcessing();