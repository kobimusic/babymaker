// AudioWorklet: raw mic samples, and the render frame of the first one, so a take can be
// placed on the film's timeline to the sample. Also reports the input peak for the meter.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false;
    this.buf = new Float32Array(4096);
    this.n = 0;
    this.peak = 0;
    this.peakN = 0;
    this.port.onmessage = (e) => {
      if (e.data === "start") {
        this.on = true;
        this.first = true;
        this.n = 0;
      } else if (e.data === "stop") {
        this.flush();
        this.on = false;
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  flush() {
    if (this.n) this.port.postMessage({ type: "data", buf: this.buf.slice(0, this.n) });
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    const len = ch ? ch.length : 128;
    for (let i = 0; ch && i < len; i++) {
      const a = Math.abs(ch[i]);
      if (a > this.peak) this.peak = a;
    }
    this.peakN += len;
    if (this.peakN >= 2048) {
      this.port.postMessage({ type: "level", peak: this.peak });
      this.peak = 0;
      this.peakN = 0;
    }
    if (this.on) {
      if (this.first) {
        this.port.postMessage({ type: "first", frame: currentFrame });
        this.first = false;
      }
      for (let i = 0; i < len; i++) {
        this.buf[this.n++] = ch ? ch[i] : 0;          // a dropped block is silence, not a gap
        if (this.n === this.buf.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor("capture", Capture);
