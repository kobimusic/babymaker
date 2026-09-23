/**
 * Standard MIDI File reader — enough to play a piece and to draw one.
 *
 * Playing needs the notes and the tempo map that turns their ticks into
 * seconds. Drawing a piano roll needs a little more, all of it already in the
 * file: which track a note came from and what the track calls itself, which
 * instrument each channel was given, and the tempo and time-signature maps
 * themselves, because bar lines are counted in ticks and only then placed in
 * time. Everything a player used before is still here under the same name.
 */
export interface MidiNote {
  note: number;
  velocity: number;   // 1..127
  channel: number;
  start: number;      // seconds
  end: number;        // seconds
  /** which of the file's tracks the note was written in */
  track: number;
}

/** A track of the file, as it describes itself. */
export interface MidiTrack {
  index: number;
  /** its own name (meta event 03), or "" when it gives none */
  name: string;
  /** how many notes start in it */
  notes: number;
}

/** A change of tempo. The first is always at tick 0: a file that sets none plays at 120 bpm. */
export interface MidiTempo { tick: number; time: number; usPerBeat: number }

/** A change of metre. The first is always at tick 0: a file that sets none is in 4/4. */
export interface MidiTimeSignature { tick: number; time: number; numerator: number; denominator: number }

/** A channel being given an instrument (a General MIDI program, 0..127). */
export interface MidiProgram { tick: number; time: number; channel: number; program: number }

export interface Midi {
  ticksPerBeat: number;
  notes: MidiNote[];
  /** seconds — when the last note ends */
  duration: number;
  /** 0: a single track · 1: several, sounding together */
  format: number;
  tracks: MidiTrack[];
  tempos: MidiTempo[];
  timeSignatures: MidiTimeSignature[];
  programs: MidiProgram[];
}

/** what a file that says nothing plays at: 120 beats a minute */
const DEFAULT_US_PER_BEAT = 500000;

/**
 * A name as text. Files do not say how their text is encoded: most are ASCII,
 * newer ones UTF-8, older ones Latin-1 — which is not valid UTF-8 the moment
 * it holds an accent, so that is how the two are told apart.
 */
function text(bytes: Uint8Array): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes).trim();
  } catch {
    return new TextDecoder("latin1").decode(bytes).trim();
  }
}

export function parseMidi(buf: ArrayBuffer): Midi {
  const v = new DataView(buf);
  const u8 = new Uint8Array(buf);
  const tag = (o: number) => String.fromCharCode(u8[o], u8[o + 1], u8[o + 2], u8[o + 3]);
  if (tag(0) !== "MThd") throw new Error("not a MIDI file");
  const division = v.getUint16(12);
  if (division & 0x8000) throw new Error("SMPTE time division is not supported");
  const ticksPerBeat = division;
  const format = v.getUint16(8);
  const nTracks = v.getUint16(10);

  // One shape for every kind of event, the fields a kind does not use left
  // at zero: they are sorted together, and an array of one shape sorts faster
  // than a union does.
  interface Ev {
    tick: number; tr: number; type: "on" | "off" | "tempo" | "metre" | "program";
    ch: number; a: number; b: number;
  }
  const events: Ev[] = [];
  const tracks: MidiTrack[] = [];

  let o = 8 + v.getUint32(4);
  for (let tr = 0; tr < nTracks && o + 8 <= buf.byteLength; tr++) {
    if (tag(o) !== "MTrk") throw new Error("expected MTrk");
    const len = v.getUint32(o + 4);
    let p = o + 8;
    const end = p + len;
    let tick = 0;
    let status = 0;
    const track: MidiTrack = { index: tr, name: "", notes: 0 };
    tracks.push(track);

    const vlq = () => {
      let n = 0;
      for (;;) { const b = u8[p++]; n = (n << 7) | (b & 0x7f); if (!(b & 0x80)) break; }
      return n;
    };

    while (p < end) {
      tick += vlq();
      let b = u8[p];
      if (b & 0x80) { status = b; p++; } // else running status

      if (status === 0xff) {                      // meta
        const type = u8[p++];
        const l = vlq();
        if (type === 0x51) {
          const us = (u8[p] << 16) | (u8[p + 1] << 8) | u8[p + 2];
          events.push({ tick, tr, type: "tempo", ch: 0, a: us, b: 0 });
        } else if (type === 0x58) {
          // the denominator is written as a power of two: 3 is an eighth
          events.push({ tick, tr, type: "metre", ch: 0, a: u8[p], b: 2 ** u8[p + 1] });
        } else if (type === 0x03 && !track.name) {
          track.name = text(u8.subarray(p, p + l));
        }
        p += l;
      } else if (status === 0xf0 || status === 0xf7) {   // sysex
        // The length is read first and added after. `p += vlq()` reads `p`
        // before vlq() has moved it past the length's own bytes, so the
        // payload was skipped short by exactly that many: the closing F7 was
        // then taken for the next delta time, and every note after a GM
        // reset — which is how most exported files open — landed seconds late.
        const payload = vlq();
        p += payload;
      } else {
        const hi = status & 0xf0, ch = status & 0x0f;
        b = u8[p];
        if (hi === 0x90 || hi === 0x80) {
          const note = u8[p], vel = u8[p + 1]; p += 2;
          const on = hi === 0x90 && vel > 0;
          events.push({ tick, tr, type: on ? "on" : "off", ch, a: note, b: vel });
        } else if (hi === 0xa0 || hi === 0xb0 || hi === 0xe0) {
          p += 2;
        } else if (hi === 0xc0) {
          events.push({ tick, tr, type: "program", ch, a: u8[p], b: 0 });
          p += 1;
        } else if (hi === 0xd0) {
          p += 1;
        } else {
          throw new Error(`unexpected status ${status.toString(16)} at ${p}`);
        }
      }
    }
    o = end;
  }

  // ticks -> seconds through the tempo map (tempo changes apply to all tracks)
  events.sort((a, b) => a.tick - b.tick || (a.type === "tempo" ? -1 : 0) - (b.type === "tempo" ? -1 : 0));
  let usPerBeat = DEFAULT_US_PER_BEAT, lastTick = 0, seconds = 0;
  const toSec = (tick: number) => seconds + ((tick - lastTick) * usPerBeat) / 1e6 / ticksPerBeat;

  const notes: MidiNote[] = [];
  const tempos: MidiTempo[] = [];
  const timeSignatures: MidiTimeSignature[] = [];
  const programs: MidiProgram[] = [];
  const open = new Map<number, MidiNote>();   // key: ch*128 + note
  for (const e of events) {
    const t = toSec(e.tick);
    if (e.type === "tempo") {
      seconds = t; lastTick = e.tick; usPerBeat = e.a;
      // a second tempo on the same tick replaces the first: only the last one is ever heard
      if (tempos.length && tempos[tempos.length - 1].tick === e.tick) tempos.pop();
      tempos.push({ tick: e.tick, time: t, usPerBeat: e.a });
      continue;
    }
    if (e.type === "metre") {
      if (timeSignatures.length && timeSignatures[timeSignatures.length - 1].tick === e.tick) timeSignatures.pop();
      timeSignatures.push({ tick: e.tick, time: t, numerator: e.a, denominator: e.b });
      continue;
    }
    if (e.type === "program") {
      programs.push({ tick: e.tick, time: t, channel: e.ch, program: e.a });
      continue;
    }
    const key = e.ch * 128 + e.a;
    if (e.type === "on") {
      const prev = open.get(key);
      if (prev) prev.end = t;                   // retrigger closes the earlier one
      const n: MidiNote = { note: e.a, velocity: e.b, channel: e.ch, start: t, end: t, track: e.tr };
      open.set(key, n);
      notes.push(n);
      tracks[e.tr].notes++;
    } else {
      const n = open.get(key);
      if (n) { n.end = t; open.delete(key); }
    }
  }
  // anything left hanging gets a short tail
  for (const n of open.values()) n.end = Math.max(n.end, n.start + 0.5);

  // Both maps open at tick 0, so whoever counts bars never has to ask what
  // held before the first change.
  if (!tempos.length || tempos[0].tick > 0) tempos.unshift({ tick: 0, time: 0, usPerBeat: DEFAULT_US_PER_BEAT });
  if (!timeSignatures.length || timeSignatures[0].tick > 0) {
    timeSignatures.unshift({ tick: 0, time: 0, numerator: 4, denominator: 4 });
  }

  notes.sort((a, b) => a.start - b.start);
  const duration = notes.reduce((m, n) => Math.max(m, n.end), 0);
  return { ticksPerBeat, notes, duration, format, tracks, tempos, timeSignatures, programs };
}
