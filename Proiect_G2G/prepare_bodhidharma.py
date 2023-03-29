# Run this script to prepare the Bodhidharma data in the current directory.
# Will download the dataset first.

import os
import shutil
import subprocess
import sys
import tempfile
import gzip
import json
import csv
import urllib.request
import zipfile

src_dir='MIDI_Files'

def die(*args):
    if len(args) == 0:
        args = ('Failed.',)
    print(*args, file=sys.stderr)
    sys.exit(1)

def log(*args):
    print(*args, file=sys.stderr)

def log_progress(*args):
    print("\r\033[2K" + " ".join(args) + " ", end="", file=sys.stderr)

tmp_dir = tempfile.mkdtemp()
def cleanup():
    shutil.rmtree(tmp_dir)
    log(f"Deleted temporary directory {tmp_dir}.")

try:
    if not os.path.exists(src_dir):
        url = "http://www.music.mcgill.ca/~cmckay/protected/Bodhidharma_MIDI.zip"
        urllib.request.urlretrieve(url, "Bodhidharma_MIDI.zip")
        with zipfile.ZipFile("Bodhidharma_MIDI.zip") as archive:
            archive.extractall()
        os.remove("Bodhidharma_MIDI.zip")
    
    if not os.path.exists(src_dir):
        die(f"{src_dir} does not exist")

    # Fix the key signatures and filenames
    dir = '01_fixed'
    if not os.path.exists(dir):
        os.mkdir(dir)

        src_files = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith('.mid'):
                    src_files.append(os.path.join(root, file))

        num_files = len(src_files)
        print(f'Found {num_files} files in {src_dir}')

        for src_file in src_files:
            fname = os.path.basename(src_file).replace('.mid', '.mid', 1)
            log_progress(fname)
            dest_file = os.path.join(dir, fname)
            subprocess.run(['python', '-m', 'scripts.fix_midi_key_signatures', src_file, dest_file], check=True)

        num_created = len(os.listdir(dir))
        print(f'Created {num_created} files in {dir}')

    # Filter the files to have 4/4 time only
    input_dir = "01_fixed"
    dir = "02_filtered"
    if not os.path.exists(dir):
        os.mkdir(dir)
        linked_files = 0

        for filename in os.listdir(input_dir):
            if filename.endswith(".mid"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(dir, filename)
                process = subprocess.run(["python", "-m", "scripts.filter_4beats", input_path], capture_output=True, text=True)
                if process.returncode == 0:
                    log_progress(filename)
                    os.link(input_path, output_path)
                    linked_files += 1
                else:
                    die()

        log()
        log(f"Linked {linked_files} files to {dir}")

    # Chop the files into 8-bar segments, save as NoteSequences
    dir = "03_chopped"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

        cmd = [
            "python",
            "-m",
            "scripts.chop_midi",
            "--bars-per-segment",
            "8",
            "--min-notes-per-segment",
            "1",
            "--merge-instruments",
            "--force-tempo",
            "60",
            "02_filtered/",
            f"{dir}/data"
        ]

        subprocess.run(cmd, check=True)

    # Separate the instrument tracks
    dir = "04_separated"
    if not os.path.exists(dir):
        os.mkdir(dir)

        instr = "all_except_drums"
        cmd = f"python -m scripts.filter_note_sequences --no-drums 03_chopped/data.tfrecord {dir}/{instr}.tfrecord"
        subprocess.check_call(cmd, shell=True)

        os.link("03_chopped/data.tfrecord", f"{dir}/all.tfrecord")

    # Make an LMDB database
    dir = "05_db"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

        for recordfile in os.listdir("04_separated"):
            if not recordfile.endswith(".tfrecord"):
                continue
            prefix = os.path.splitext(recordfile)[0]
            tmp_db_name = f"{prefix}.db"
            tmp_db_path = os.path.join(tmp_dir, tmp_db_name)
            subprocess.run(["python", "-m", "scripts.tfrecord_to_lmdb", f'04_separated/{recordfile}', tmp_db_path], check=True)
            os.remove(f"{tmp_db_path}-lock")
            for f in os.listdir(tmp_dir):
                if f.startswith(prefix):
                    shutil.move(os.path.join(tmp_dir, f), dir)

    dir = 'final'

    if not os.path.exists(dir):
        os.mkdir(dir)

    for filename in os.listdir('04_separated'):
        src_path = os.path.join('04_separated', filename)
        dst_path = os.path.join(dir, filename)
        os.link(src_path, dst_path)

    for filename in os.listdir('05_db'):
        src_path = os.path.join('05_db', filename)
        dst_path = os.path.join(dir, filename)
        os.link(src_path, dst_path)

    bodh_meta = {}
    with open('recordings_key.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            filename, song_name, artist, genre = row
            bodh_meta[os.path.splitext(filename)[0] + '.mid'] = {
                'song_name': song_name,
                'artist': artist,
                'genre': genre
            }

    with gzip.open('03_chopped/data_meta.json.gz', 'rt') as f:
        data = json.load(f)

    data_dict = {}
    key_len = len(str(len(data) - 1))
    for i, item in enumerate(data):
        item.update(bodh_meta[item['filename']])
        key = str(i).zfill(key_len)
        data_dict[key] = item

    with gzip.open(os.path.join(dir, 'meta.json.gz'), 'wt') as f:
        json.dump(data_dict, f, separators=(',', ':'))

    print('Done.')
finally:
    cleanup()