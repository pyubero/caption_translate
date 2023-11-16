import re
from tqdm import tqdm
from deep_translator import GoogleTranslator


def translate_captions(captions, target, flag="*"):
    all_captions = flag.join(captions)

    translated = GoogleTranslator(
        source='auto',
        target=target).translate(all_captions)

    return translated.split(flag)


def export_srt(filename, start, end, captions):
    assert len(start) == len(captions)
    assert len(end) == len(captions)

    with open(filename, "w+") as file:
        for ii, caption in enumerate(captions):
            if caption=="":
                continue

            _ = file.write(f"{ii}\n")
            _ = file.write(f"{start[ii]} --> {end[ii]}\n")
            _ = file.write(caption+"\n")
            _ = file.write("\n")


# --- Tunable constants ---
SRTFILE = "video_deu.srt"
LANG_IN = "deu"
LANG_OUT = ["es", "en"]

# Other parameters
start, end, captions = [], [], []
srtfilename =  ".".join(SRTFILE.split(".")[:-1])

# Load captions in current language
with open(SRTFILE, "r") as file:
    for line in tqdm(file.readlines()):

        if "-->" in line:
            times = line.replace("\n","").split("-->")
            start.append(times[0].replace(" ",""))
            end.append(times[1].replace(" ",""))

        if re.search('[a-zA-Z]', line):
            captions.append(line.replace("\n",""))

# Export captions in other languages
for lang in LANG_OUT:
    outputfilename = f"{srtfilename}_{lang}.srt"
    captions_tr = translate_captions(captions, lang, flag="*")
    export_srt(outputfilename, start, end, captions_tr)
    print(f"Captions exported in {lang}.")
