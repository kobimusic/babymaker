"""The narration for both videos, in Nathan's words.

Each line is either a string, or {"cap": what the captions show, "say": what the voice
reads} when the two differ (acronyms, numbers). Line ids are "<section>.<n>".
"""

SHORT = {
    "open": [
        "So I wanted to take three instruments and make a new one out of them. Not a crossfade, but a note that is part of each.",
        "The attack, the tone and even the decay would all land in the middle.",
    ],
    "pianos": [
        "It works on soundfonts too. This is Satie, with the piano morphing between three soundfonts as it plays.",
    ],
    "band": [
        "It is also able to do a full arrangement, where each instrument is a baby.",
    ],
    "end": [
        "I called it babymaker, and the long video goes over how it works.",
    ],
}

LONG = {
    "cold": [
        "This is Mozart, on an instrument that does not exist anywhere.",
        "It is part flute, part violin and part harp. I am able to drag it around while the song is playing, and the sound follows it.",
    ],
    "why": [
        "So why would I want this? I train music transcription models, which are models that listen to audio and turn it into notes.",
        "Most of my training audio is synthetic. I take MIDI files and render them through soundfonts, so I already know what the notes are.",
        "The problem is there are only so many soundfonts. I think a model can end up learning those specific pianos, instead of what a piano sounds like in general.",
        "If I could make new instruments out of the ones I already have, I would be able to get way more timbres without recording anything.",
    ],
    "demo": [
        "Here is the demo from my site. Three instruments sit on the dots, and each one has a single note.",
        "The pink dot is the playhead. If I drag it onto the flute, you hear the flute sample with no changes.",
        "Same thing with the violin.",
        "The harp is plucked, so as I get closer to it the notes actually get shorter. The decay is being interpolated too, not only the tone.",
        "In the middle, it is not three sounds stacked on top of each other (which is what a crossfade would give you). It is one note.",
    ],
    "gap": [
        "A sample library could give you a flute, a violin and a harp. What it will not give you is a mix of them that sounds like its own instrument.",
        "If I play all three at the same time, you get three sounds and not one. Here are their levels on top of each other, with three attacks, three decays and three different lengths.",
        "So to get a single sound out of the three, I have to take the recordings apart first.",
    ],
    "transform": [
        {"cap": "First, I get out of the time domain. I cut the signal into overlapping windows, and each window goes through a lapped cosine transform, called the MCLT.",
         "say": "First, I get out of the time domain. I cut the signal into overlapping windows, and each window goes through a lapped cosine transform, called the M C L T."},
        "It is basically a cosine transform and a sine transform together, so each frequency bin has a magnitude and a phase.",
        {"cap": "It is also critically sampled. There are 1024 bins per frame and a hop of 512, and the frames add back up to the original signal with no error.",
         "say": "It is also critically sampled. There are a thousand and twenty four bins per frame and a hop of five hundred and twelve, and the frames add back up to the original signal with no error."},
    ],
    "split": [
        "Then I split the log spectrum of each frame into three parts.",
        "The first part is the level, which is how loud that frame is in decibels.",
        "The second is the envelope, which is the smooth curve the harmonics sit on. I think of it as the timbre of the instrument, and I find it with an iterated cepstral smoothing.",
        "If you smooth the spectrum one time, the curve goes right through the harmonics. So I take the max of the spectrum and the curve, and smooth it again, and again. The envelope climbs a bit each time, until it sits on top of the peaks.",
        "Whatever is left over is the fine structure. That is the harmonic comb, plus the noise between the partials.",
    ],
    "align": [
        "The three notes are not the same length, and their attacks and decays do not line up. If I averaged them like this, each transient would get smeared.",
        "So I warp each source onto the first one with dynamic time warping. The features are the level and the first few orders of the cepstrum, since those describe how the note changes over time.",
        "Here is the cost matrix between the flute and the violin, with the cheapest path through it.",
        "I use that path as a map (this frame of the flute goes with that frame of the violin). Then I smooth it, so the warp never suddenly speeds up or slows down.",
    ],
    "average": [
        "Now for the weights. I picked forty five percent flute, thirty five violin, and twenty harp.",
        "The output timeline is the weighted average of the aligned timelines. For instance, a slow attack and a fast attack would average out to a medium attack, instead of being crossfaded.",
        "The level is averaged in decibels, so that the decay rates get averaged too.",
        "The envelope is averaged in the log domain. If I averaged amplitudes, the loudest source would basically win the timbre. With logs, all three of them count.",
        "The fine structure is averaged as amplitude to the power of zero point six. I chose that because raw amplitude loses the quiet partials, and decibels care too much about the silence between them.",
        "The Python version has one more step for the envelope. The formant band uses optimal transport, so a peak at five hundred hertz and a peak at one kilohertz meet at around seven hundred, instead of turning into a pair of half height bumps.",
    ],
    "synth": [
        "So now there is a magnitude for each bin of each frame. But there is no phase, and the wrong phase is what turns a note into mush.",
        {"cap": "So instead of making up a phase, I borrow one. I warp each source in the time domain with WSOLA, where each block slides to the spot that best continues the block before it. Then all three sources line up sample for sample, and I can mix them.",
         "say": "So instead of making up a phase, I borrow one. I warp each source in the time domain with W sola, where each block slides to the spot that best continues the block before it. Then all three sources line up sample for sample, and I can mix them."},
        "The phase of that mix is already close. A couple rounds of Griffin-Lim, back to audio and forward again, clean up the rest.",
    ],
    "result": [
        "So three recordings go in, and one instrument comes out. It has one attack, one decay, and a timbre that is not any of the three.",
        "On my site, it is a TypeScript port running in a web worker. A new morph takes about a hundred milliseconds.",
    ],
    "pianos": [
        "Okay, now the fun part. This is the first Gymnopedie by Satie.",
        "Each note gets rebuilt at wherever the playhead is when that note starts. The first set is normal grand pianos from different soundfonts.",
        "Now the nostalgia set. There is the Microsoft GS wavetable piano (the one Windows used for MIDI), a Fairlight piano, and the piano from Mario Kart DS.",
        {"cap": "And the weird set, with FM pianos from a Yamaha FB-01 and an OPL4 chip, against a normal grand.",
         "say": "And the weird set, with FM pianos from a Yamaha F B zero one and an O P L four chip, against a normal grand."},
        "None of the pianos you hear in the middle of the triangle exist in any soundfont.",
    ],
    "band": [
        {"cap": "It also works on a full arrangement. This is DOTABATA, from the Nena MIDI collection.",
         "say": "It also works on a full arrangement. This is Dotabata, from the Nena MIDI collection."},
        "Each track is morphed, including the brass, the slap bass, the organ and the drums. Each one is a baby of the same instrument from three soundfonts.",
        "The playhead moves the full band at the same time. Here it goes from FluidR3 to the old Windows synth, and then into an FM chip.",
        "For the dataset, it would not move like this. Each track would get its own random point, so each render is a slightly different band.",
    ],
    "outro": [
        "The version on my site skips the optimal transport step, so that is the next thing I want to port.",
        "The code is on GitHub, with the Python version, the TypeScript port and the demo. Thanks for watching.",
    ],
}


def lines(script: dict) -> dict:
    """{id: {"cap", "say"}} in order."""
    out = {}
    for sec, ls in script.items():
        for i, l in enumerate(ls):
            cap, say = (l, l) if isinstance(l, str) else (l["cap"], l["say"])
            out[f"{sec}.{i}"] = {"cap": cap, "say": say}
    return out


if __name__ == "__main__":
    import sys
    which = LONG if (len(sys.argv) < 2 or sys.argv[1] == "long") else SHORT
    sec = None
    for k, v in lines(which).items():
        if sec and k.split(".")[0] != sec:
            print()
        sec = k.split(".")[0]
        print(v["cap"])
