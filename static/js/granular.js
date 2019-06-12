window.AudioContext = window.AudioContext || window.webkitAudioContext || window.mozAudioContext || window.oAudioContext;

function loadAudio(filename) {
	var request = new XMLHttpRequest();
	request.open('GET', filename, true);
	request.responseType = "arraybuffer";
	request.onload = function () {
		window.granulizer = new Granulizer(request.response);
		console.log('download complete');
	};
	request.send();
}

function playAgain() {
	window.granulizer.playGrain(Math.random() * 60.0 * 10 - 0);
}

class Granulizer {
	constructor(response) {
		this.response = response;
		this.audioIsRunning = false;
	}

	startAudio() {
		console.log('starting audio');
		this.context = new AudioContext();
		this.master = this.context.createGain();
		this.master.connect(this.context.destination);

		let that = this;
		this.context.decodeAudioData(this.response, function (buffer) {
			console.log('has loaded');
			that.buffer = buffer;
		}, function () {
			console.log('loading failed');
		});
	}

	playGrain(position) {
		if (!this.audioIsRunning) {
			this.startAudio();
			this.audioIsRunning = true;
			return;
		}
		else if (this.buffer === undefined) {
			// is still decoding
			return;
		}

		let source = this.context.createBufferSource();
		source.playbackRate.value = source.playbackRate.value;
		source.buffer = this.buffer;

		let gain = this.context.createGain();
		source.connect(gain);
		gain.connect(this.master);

		let now = this.context.currentTime;
		let amp = 1.0;
		let attack = 0.1;
		let release = 0.5;
		let sustain = 0.5;

		source.start(now, position, attack + sustain + release);
		gain.gain.setValueAtTime(0.0, now);
		gain.gain.linearRampToValueAtTime(amp, now + attack);
		gain.gain.linearRampToValueAtTime(amp, now + attack + sustain);
		gain.gain.linearRampToValueAtTime(0, now + (attack + sustain + release));

		source.stop(now + attack + sustain + release + 0.1);
		let tms = (attack + sustain + release) * 1000;
		setTimeout(function () {
			gain.disconnect();
		}, tms + 200);
	}
}