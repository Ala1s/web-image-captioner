import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 299;
const INCEPTION_PATH = 'https://breyman.ru/ala1s/inception/model.json';
const CAPTION_PATH = 'https://breyman.ru/ala1s/mymodel/model.json';
const IDX2WORD_PATH = 'https://breyman.ru/ala1s/idx2word.json';
const WORD2IDX_PATH = 'https://breyman.ru/ala1s/word2idx.json';
const TRANSLATOR_PATH = "https://translate.yandex.net/api/v1.5/tr.json/translate?";
var transl = false;
var algo = "BEAM";
var language = 'ru';
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.action === 'ARGMAX') {
    algo='ARGMAX';
  }
  if (message && message.action === 'BEAM') {
    algo='BEAM';
  }
  if (message && message.action === 'TRANSLATE') {
    transl = true;
  }
  if (message && message.action === 'DONTTRANSLATE') {
    transl = false;
  }
  if (message && message.payload && message.action === 'NEWLANG') {
    language=message.payload;
  }
});

class BackgroundOps {

  constructor() {
    this.imageReqs= {};
    this.linkListener();
    this.loadInception();
    this.loadDictionaries();
    this.loadCaptionModel();
    this.timer = setTimeout(() => {this.clearRequests},300000);

  }

  clearRequests(){
  	console.log("cleared");
  	this.imageReqs= {};
  	this.timer = setTimeout(() => {this.clearRequests},300000);

  }

  linkListener() {
    chrome.webRequest.onCompleted.addListener(req => {
      if (req && req.tabId > 0) {
        this.imageRequests[req.url] =  req;
        this.processImage(req.url);
      }
    }, { urls: ["<all_urls>"], types: ["image", "object"] });
  }

  async loadInception() {
    console.log('Loading InceptionV3...');
    const startTime = performance.now();
    this.inception = await tf.loadLayersModel(INCEPTION_PATH);
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
    this.word2idx = out;
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
    	startWord.shift();
      startWord.pop();
      return startWord.join(' ');
    })
    return argmaxCaption;
  }

async beamPredict(imgElement, beamIdx){
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
      finalCap.shift();
      console.log(finalCap.join(' '));
    	return finalCap.join(' ');
    });
    return beamCaption;

}
 async translate(caption, trlang){
 	let request = new XMLHttpRequest();
	let apikey = 'trnsl.1.1.20190411T083848Z.2884e28c5b508a68.880ce05da65d680d69dc664112a59c4262dd7fb2';
	let lang = "en-"+trlang;
	let host = "https://translate.yandex.net/api/v1.5/tr.json/translate?";

	let params = 'key=' + encodeURIComponent(apikey) +
  	'&text=' + encodeURIComponent(caption) +
  	'&lang=' + encodeURIComponent(lang)+
  	'&format=plain';
	request.open("GET", host + params, true);

	let translation;
	request.onload = () => {
		if (request.status >= 200 && request.status < 400){
			let data = JSON.parse(request.response);
			translation = data.text;
		}
	};

	request.send();
	if(translation){
		return translation
	}
 }

  async processImage(src) {

    if (!this.inception || !this.captionModel) {
      console.log('Model not loaded yet, delaying...');
      setTimeout(() => { this.processImage(src) }, 5000);
      return;
    }
    if(abort){
    	return;
    }
    var meta = this.imageReqs[src];
    if (meta && meta.tabId) {
      if (!meta.predictions) {
        const img = await this.loadImage(src);
        if (img) {
        	let cap;
        	if (algo=="BEAM"){
        		cap = await this.beamPredict(img,3);
        	}
        	else {
        		cap = await this.predict(img);
        	}
        	if (transl){
        		let trnsl = await this.translate(cap, language);
        		meta.predictions = trnsl;
        	}
        	else {
        		meta.predictions = cap;
        	}
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

var bg = new BackgroundOps();