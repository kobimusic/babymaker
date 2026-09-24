# Voiceover script

Written by build.py from video/script.py. The videos have no narration, only the music and the instrument sounds (plus silent copies), so this is the script to record over them. The captions in `out/*.srt` follow the same timing.

## Short (30 s)

`babymaker-short.mp4`, 30.0 s. The picture is timed for a read at about 2.8 words a second, so each line has room. Times are where each line starts.

| time | on screen | line |
|---|---|---|
| 0:00.2 | title | I made a new instrument out of three others. Not a crossfade. |
| 0:05.3 | tri_short | The attack, the tone and the decay all land in the middle. |
| 0:10.3 | pianos_short | It works on soundfonts too, like Satie on three pianos. |
| 0:15.9 | band_short | Or a full arrangement, where each instrument is a baby. |
| 0:20.3 | speed | No neural network, only DCTs. A new note takes about thirty milliseconds. |

## Long (explained)

`babymaker-explained.mp4`, 683.5 s. The picture is timed for a read at about 2.8 words a second, so each line has room. Times are where each line starts.

| time | on screen | line |
|---|---|---|
| 0:01.0 | cold | This is Mozart, on an instrument that does not exist anywhere. |
| 0:05.7 | cold | It is part flute, part violin and part harp. I am able to drag it around while the song is playing, and the sound follows it. |
| 0:21.7 | S0Why | So why would I want this? I train music transcription models, which are models that listen to audio and turn it into notes. |
| 0:30.6 | S0Why | Most of my training audio is synthetic. I take MIDI files and render them through soundfonts, so I already know what the notes are. |
| 0:40.0 | S0Why | The problem is there are only so many soundfonts. I think a model can end up learning those specific pianos, instead of what a piano sounds like in general. |
| 0:51.1 | S0Why | If I could make new instruments out of the ones I already have, I would be able to get way more timbres without recording anything. |
| 1:02.4 | demo | Here is the demo from my site. Three instruments sit on the dots, and each one has a single note. |
| 1:10.3 | demo | The pink dot is the playhead. If I drag it onto the flute, you hear the flute sample with no changes. |
| 1:21.1 | demo | Same thing with the violin. |
| 1:26.3 | demo | The harp is plucked, so as I get closer to it the notes actually get shorter. The decay is being interpolated too, not only the tone. |
| 1:39.3 | demo | In the middle, it is not three sounds stacked on top of each other (which is what a crossfade would give you). It is one note. |
| 1:53.7 | S1Gap | A sample library could give you a flute, a violin and a harp. What it will not give you is a mix of them that sounds like its own instrument. |
| 2:05.2 | S1Gap | If I play all three at the same time, you get three sounds and not one. Here are their levels on top of each other, with three attacks, three decays and three different lengths. |
| 2:18.1 | S1Gap | So to get a single sound out of the three, I have to take the recordings apart first. |
| 2:26.4 | S2Transform | First, I get out of the time domain. I cut the signal into overlapping windows, and each window goes through a lapped cosine transform, called the MCLT. |
| 2:37.9 | S2Transform | It is basically a cosine transform and a sine transform together, so each frequency bin has a magnitude and a phase. |
| 2:46.2 | S2Transform | It is also critically sampled. There are 1024 bins per frame and a hop of 512, and the frames add back up to the original signal with no error. |
| 3:00.9 | S3Split | Then I split the log spectrum of each frame into three parts. |
| 3:05.9 | S3Split | The first part is the level, which is how loud that frame is in decibels. |
| 3:12.1 | S3Split | The second is the envelope, which is the smooth curve the harmonics sit on. I think of it as the timbre of the instrument, and I find it with an iterated cepstral smoothing. |
| 3:24.6 | S3Split | If you smooth the spectrum one time, the curve goes right through the harmonics. So I take the max of the spectrum and the curve, and smooth it again, and again. The envelope climbs a bit each time, until it sits on top of the peaks. |
| 3:41.8 | S3Split | Whatever is left over is the fine structure. That is the harmonic comb, plus the noise between the partials. |
| 3:50.5 | S4Align | The three notes are not the same length, and their attacks and decays do not line up. If I averaged them like this, each transient would get smeared. |
| 4:01.3 | S4Align | So I warp each source onto the first one with dynamic time warping. The features are the level and the first few orders of the cepstrum, since those describe how the note changes over time. |
| 4:14.5 | S4Align | Here is the cost matrix between the flute and the violin, with the cheapest path through it. |
| 4:21.4 | S4Align | I use that path as a map (this frame of the flute goes with that frame of the violin). Then I smooth it, so the warp never suddenly speeds up or slows down. |
| 4:35.2 | S5Average | Now for the weights. I picked forty five percent flute, thirty five violin, and twenty harp. |
| 4:41.7 | S5Average | The output timeline is the weighted average of the aligned timelines. For instance, a slow attack and a fast attack would average out to a medium attack, instead of being crossfaded. |
| 4:53.5 | S5Average | The level is averaged in decibels, so that the decay rates get averaged too. |
| 4:59.3 | S5Average | The envelope is averaged in the log domain. If I averaged amplitudes, the loudest source would basically win the timbre. With logs, all three of them count. |
| 5:09.7 | S5Average | The fine structure is averaged as amplitude to the power of zero point six. I chose that because raw amplitude loses the quiet partials, and decibels care too much about the silence between them. |
| 5:22.6 | S5Average | The Python version has one more step for the envelope. The formant band uses optimal transport, so a peak at five hundred hertz and a peak at one kilohertz meet at around seven hundred, instead of turning into a pair of half height bumps. |
| 5:40.2 | S6Synth | So now there is a magnitude for each bin of each frame. But there is no phase, and the wrong phase is what turns a note into mush. |
| 5:50.9 | S6Synth | So instead of making up a phase, I borrow one. I warp each source in the time domain with WSOLA, where each block slides to the spot that best continues the block before it. Then all three sources line up sample for sample, and I can mix them. |
| 6:09.2 | S6Synth | The phase of that mix is already close. A couple rounds of Griffin-Lim, back to audio and forward again, clean up the rest. |
| 6:19.4 | S7Result | So three recordings go in, and one instrument comes out. It has one attack, one decay, and a timbre that is not any of the three. |
| 6:29.4 | S7Result | And it is fast enough to run while you drag the dot around. The reason it is that fast is mostly one transform. |
| 6:39.7 | S7bDCT | Pretty much the whole algorithm is the DCT, and it shows up three times. |
| 6:45.5 | S7bDCT | The MDCT that cuts the audio into frames is a DCT. The cepstrum that finds the envelope is a DCT of the log spectrum. And going back to audio is a DCT again. |
| 6:58.0 | S7bDCT | It is lossless. If I take this flute into the MDCT and straight back out, the worst error is 4 × 10⁻¹⁶, which is basically float rounding. |
| 7:10.9 | S7bDCT | It also packs the shape of a spectrum into very few numbers. The first 20 cepstral coefficients, out of 1024, already hold two thirds of this frame's log spectrum. Those 20 numbers are the envelope, and the rest is the harmonic comb. |
| 7:28.1 | S7bDCT | And it is cheap. A DCT of 1024 points takes about 4 microseconds, because it runs through an FFT. A four second note is around 700 frames, so it barely costs anything. |
| 7:42.0 | S7bDCT | There is also no training. None of this is learned, so it works on any three recordings I give it. |
| 7:51.1 | S7cSpeed | So how does that compare? Most of the other ways to morph between instruments are neural networks. |
| 7:58.0 | S7cSpeed | NSynth, from Google Magenta in 2017, could morph between instruments with a WaveNet autoencoder. One four second note took around eighteen minutes on a GPU. |
| 8:07.6 | S7cSpeed | GANSynth and RAVE got that to milliseconds. But they have to be trained on a dataset first, and GANSynth cannot take your own samples at all. |
| 8:17.7 | S7cSpeed | babymaker makes the same four second note in about thirty milliseconds on my GPU, or a fifth of a second on the CPU. That is with no training and no dataset, and it even runs in a browser. |
| 8:34.4 | pianos_long | Okay, now the fun part. This is the first Gymnopedie by Satie. |
| 8:39.5 | pianos_long | Each note gets rebuilt at wherever the playhead is when that note starts. The first set is normal grand pianos from different soundfonts. |
| 9:10.0 | pianos_long | Now the nostalgia set. There is the Microsoft GS wavetable piano (the one Windows used for MIDI), a Fairlight piano, and the piano from Mario Kart DS. |
| 9:44.6 | pianos_long | And the weird set, with FM pianos from a Yamaha FB-01 and an OPL4 chip, against a normal grand. |
| 10:04.6 | pianos_long | None of the pianos you hear in the middle of the triangle exist in any soundfont. |
| 10:14.4 | band_long | It also works on a full arrangement. This is DOTABATA, from the Nena MIDI collection. |
| 10:20.6 | band_long | Each track is morphed, including the brass, the slap bass, the organ and the drums. Each one is a baby of the same instrument from three soundfonts. |
| 10:30.9 | band_long | The playhead moves the full band at the same time. Here it goes from FluidR3 to the old Windows synth, and then into an FM chip. |
| 10:52.1 | band_long | For the dataset, it would not move like this. Each track would get its own random point, so each render is a slightly different band. |
| 11:10.0 | S8Outro | The version on my site skips the optimal transport step, so that is the next thing I want to port. |
| 11:18.9 | kobimusic | Thanks for watching. |
