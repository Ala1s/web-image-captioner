var checkBox = document.getElementById('translate');
checkBox.onChange = () => {
	if(checkBox.checked){
		chrome.runtime.sendMessage({action: 'TRANSLATE'});
	}
	else{
		chrome.runtime.sendMessage({action: 'DONTTRANSLATE'});
	}
}