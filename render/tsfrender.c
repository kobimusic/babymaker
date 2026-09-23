/* Per-track event-list -> WAV renderer built on TinySoundFont (C).
 *
 * Usage:
 *   tsfrender <sf2> <events.txt> <out.wav> <sample_rate> <total_dur_s> <gain_db> [mono]
 *   tsfrender --list <sf2>            (prints "bank preset name" per preset)
 *
 * Events file: one event per line, timed events sorted by time (seconds
 * from render start). Untimed lines apply immediately in file order.
 *   K <chan> <bank> <preset>      select bank/preset (bank 128 = drum kits)
 *   G <chan> <program>            select GM melodic program (bank 0 + fallbacks)
 *   D <chan> <program>            select GM drum kit (bank 128 + fallbacks)
 *   R <chan> <semitones>          pitch-bend range for the channel
 *   P <chan> <pan>                pan in [-1, 1]
 *   V <chan> <volume>             channel volume in [0, 1]
 *   N <t> <chan> <pitch> <vel01>  note on (velocity 0..1)
 *   F <t> <chan> <pitch>          note off
 *   B <t> <chan> <wheel>          pitch wheel 0..16383 (8192 = centre)
 *   X <t> <chan> <cc> <value>     MIDI control change
 * Output is a 32-bit float WAV (stereo interleaved unless "mono" is given).
 * The peak absolute sample value is printed on stdout ("peak %f").
 */
#define TSF_IMPLEMENTATION
#include "tsf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void write_wav_f32(const char *path, const float *buf, long frames, int ch, int sr)
{
    long n = frames * ch;
    float peak = 0.f;
    for (long i = 0; i < n; i++) { float a = fabsf(buf[i]); if (a > peak) peak = a; }
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    unsigned int data_bytes = (unsigned int)(n * 4);
    unsigned int riff = 4 + (8 + 16) + (8 + data_bytes);
    unsigned short fmt = 3, nch = (unsigned short)ch, ba = (unsigned short)(4 * ch), bits = 32;
    unsigned int sr_u = (unsigned int)sr, br = sr_u * ba, fmtlen = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmtlen, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&nch, 2, 1, f); fwrite(&sr_u, 4, 1, f);
    fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_bytes, 4, 1, f);
    fwrite(buf, 4, n, f);
    fclose(f);
    printf("peak %.6f\n", peak);
}

static void advance(tsf *f, float *buf, long *pos, long target, long total, int ch)
{
    if (target > total) target = total;
    if (target > *pos) {
        tsf_render_float(f, buf + (*pos) * ch, (int)(target - *pos), 0);
        *pos = target;
    }
}

int main(int argc, char **argv)
{
    if (argc == 3 && !strcmp(argv[1], "--list")) {
        tsf *f = tsf_load_filename(argv[2]);
        if (!f) { fprintf(stderr, "failed to load sf2 %s\n", argv[2]); return 1; }
        for (int i = 0; i < f->presetNum; i++)
            printf("%d %d %s\n", f->presets[i].bank, f->presets[i].preset, f->presets[i].presetName);
        tsf_close(f);
        return 0;
    }
    if (argc < 7) {
        fprintf(stderr, "usage: %s sf2 events out.wav sr total_dur gain_db [mono]\n       %s --list sf2\n", argv[0], argv[0]);
        return 1;
    }
    const char *sf2_path = argv[1], *ev_path = argv[2], *out_path = argv[3];
    int sr = atoi(argv[4]);
    double total_dur = atof(argv[5]);
    float gain_db = (float)atof(argv[6]);
    int mono = (argc > 7 && !strcmp(argv[7], "mono"));
    int ch = mono ? 1 : 2;

    tsf *f = tsf_load_filename(sf2_path);
    if (!f) { fprintf(stderr, "failed to load sf2 %s\n", sf2_path); return 1; }
    tsf_set_output(f, mono ? TSF_MONO : TSF_STEREO_INTERLEAVED, sr, gain_db);
    tsf_set_max_voices(f, 256);

    FILE *ev = fopen(ev_path, "r");
    if (!ev) { fprintf(stderr, "failed to open events %s\n", ev_path); return 1; }

    long total_samples = (long)(total_dur * sr);
    if (total_samples < 1) total_samples = 1;
    float *buf = (float *)calloc((size_t)total_samples * ch, sizeof(float));
    if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }
    long pos = 0;
    char line[512];

    while (fgets(line, sizeof line, ev)) {
        char type = line[0];
        int chan, a, b;
        float x;
        double t;
        switch (type) {
        case 'K':
            if (sscanf(line + 1, "%d %d %d", &chan, &a, &b) == 3) {
                if (!tsf_channel_set_bank_preset(f, chan, a, b))
                    tsf_channel_set_presetnumber(f, chan, b, a == 128);
            }
            break;
        case 'G':
            if (sscanf(line + 1, "%d %d", &chan, &a) == 2) tsf_channel_set_presetnumber(f, chan, a, 0);
            break;
        case 'D':
            if (sscanf(line + 1, "%d %d", &chan, &a) == 2) tsf_channel_set_presetnumber(f, chan, a, 1);
            break;
        case 'R':
            if (sscanf(line + 1, "%d %f", &chan, &x) == 2) tsf_channel_set_pitchrange(f, chan, x);
            break;
        case 'P':
            if (sscanf(line + 1, "%d %f", &chan, &x) == 2) tsf_channel_set_pan(f, chan, (x + 1.f) * 0.5f);
            break;
        case 'V':
            if (sscanf(line + 1, "%d %f", &chan, &x) == 2) tsf_channel_set_volume(f, chan, x);
            break;
        case 'N':
            if (sscanf(line + 1, "%lf %d %d %f", &t, &chan, &a, &x) == 4) {
                advance(f, buf, &pos, (long)(t * sr), total_samples, ch);
                tsf_channel_note_on(f, chan, a, x);
            }
            break;
        case 'F':
            if (sscanf(line + 1, "%lf %d %d", &t, &chan, &a) == 3) {
                advance(f, buf, &pos, (long)(t * sr), total_samples, ch);
                tsf_channel_note_off(f, chan, a);
            }
            break;
        case 'B':
            if (sscanf(line + 1, "%lf %d %d", &t, &chan, &a) == 3) {
                advance(f, buf, &pos, (long)(t * sr), total_samples, ch);
                tsf_channel_set_pitchwheel(f, chan, a);
            }
            break;
        case 'X':
            if (sscanf(line + 1, "%lf %d %d %d", &t, &chan, &a, &b) == 4) {
                advance(f, buf, &pos, (long)(t * sr), total_samples, ch);
                tsf_channel_midi_control(f, chan, a, b);
            }
            break;
        default:
            break;
        }
    }
    fclose(ev);
    advance(f, buf, &pos, total_samples, total_samples, ch);

    write_wav_f32(out_path, buf, total_samples, ch, sr);
    tsf_close(f);
    free(buf);
    return 0;
}
