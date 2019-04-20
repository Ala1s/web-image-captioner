let request = new XMLHttpRequest();
let apikey = 'trnsl.1.1.20190411T083848Z.2884e28c5b508a68.880ce05da65d680d69dc664112a59c4262dd7fb2';
let host = "https://translate.yandex.net/api/v1.5/tr.json/getLangs?";
let params = 'key=' + encodeURIComponent(apikey) + '&ui=en';
request.open("GET", host + params, true);
var checkBox = document.getElementById('translate');
var algo = document.getElementById('algorithm');
var lang = document.getElementById('language');
chrome.storage.sync.get('translateOption', (data) => {
    if (data.translateOption=='TRANSLATE'){
    	checkBox.checked=true;
    }
    else{
    	checkBox.checked=false;
    }
    checkBox.onchange();
});
chrome.storage.sync.get('algorithmOption', (data) => {
    if (data.algorithmOption=='ARGMAX'){
    	algo.selectedIndex=0;
    }
    else{
    	algo.selectedIndex=1;
    }
    algo.onchange();
});

request.onload = () => {
	if (request.status != 401 && request.status!=402){
		let langs = JSON.parse(request.response).langs;
		Object.keys(langs).forEach((key)=>{
			var el = document.createElement("option");
			el.text = langs[key];
			el.value = key;
			if(el.text!="English"){
				lang.appendChild(el);
			}
		})
	}
	chrome.storage.sync.get('langOption', (data) => {
	if(data.langOption){
    lang.value = data.langOption;
    lang.onchange();
	}
});
};
request.send();
checkBox.onchange = () => {
	if(checkBox.checked){
		lang.disabled = false;
		chrome.runtime.sendMessage({action: 'TRANSLATE'});
		chrome.storage.sync.set({ translateOption: 'TRANSLATE'});
	}
	else{
		lang.disabled = true;
		chrome.runtime.sendMessage({action: 'DONTTRANSLATE'});
		chrome.storage.sync.set({ translateOption: 'DONTTRANSLATE'});
	}
}
algo.onchange = () => {
if(algo.options[algo.selectedIndex].value == 'am'){
	chrome.runtime.sendMessage({action: 'ARGMAX'});
	chrome.storage.sync.set({ algorithmOption: 'ARGMAX'});
}
else{
	chrome.runtime.sendMessage({action: 'BEAM'});
	chrome.storage.sync.set({ algorithmOption: 'BEAM'});
	}
}
lang.onchange = () => {
	chrome.runtime.sendMessage({action:'NEWLANG', payload:lang.options[lang.selectedIndex].value});
	chrome.storage.sync.set({ langOption: lang.options[lang.selectedIndex].value});
}