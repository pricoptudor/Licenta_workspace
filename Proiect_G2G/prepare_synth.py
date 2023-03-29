import os
import shutil
import subprocess
import sys
import tempfile
import gzip
import json
import random

def die(*args):
    if len(args) == 0:
        args = ('Failed.',)
    print(*args, file=sys.stderr)
    sys.exit(1)


def log(*args):
    print(*args, file=sys.stderr)


def log_progress(*args):
    print("\r\033[2K" + " ".join(args) + " ", end="", file=sys.stderr)


if len(sys.argv) != 2:
    die('Expected exactly one argument: the working directory')
working_dir = sys.argv[1]

os.chdir(working_dir)

tmp_dir = tempfile.mkdtemp()
def cleanup():
    shutil.rmtree(tmp_dir)
    log(f"Deleted temporary directory {tmp_dir}.")
    
try:
    data_dir = 'fixed'
    if not os.path.isdir(data_dir):
        die(f"{data_dir} does not exist.")
    
    log(f"Preparing data in {os.getcwd()}")

    dir = '02_chopped'
    if not os.path.exists(dir):
        os.mkdir(dir)
        try:
            cmd = [
                'python',
                '-m',
                'scripts.chop_midi',
                '--bars-per-segment',
                '8',
                '--skip-bars',
                '2',
                '--min-notes-per-segment',
                '1',
                '--merge-instruments',
                '--force-tempo',
                '60',
                data_dir + '/',
                dir + '/data'
            ]
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            die("Command failed: " + " ".join(cmd) + "!" + str(e))

    dir = '03_separated'
    if not os.path.exists(dir):
        os.mkdir(dir)
        try:
            for instr in ['Bass', 'Piano', 'Guitar', 'Strings', 'Drums']:
                cmd = [
                    'python',
                    '-m',
                    'scripts.filter_note_sequences',
                    '--instrument-re',
                    f'^BB {instr}$',
                    '02_chopped/data.tfrecord',
                    f'{dir}/{instr}.tfrecord'
                ]
                subprocess.check_call(cmd)
            cmd = [
                'python',
                '-m',
                'scripts.filter_note_sequences',
                '--no-drums',
                '02_chopped/data.tfrecord',
                f'{dir}/all_except_drums.tfrecord'
            ]
            subprocess.check_call(cmd)
            source_file = '02_chopped/data.tfrecord'
            target_file = os.path.join(dir, 'all.tfrecord')

            os.link(source_file, target_file)
        except subprocess.CalledProcessError:
            die("Command failed: " + " ".join(cmd))
        
    dir = '04_db'
    if not os.path.exists(dir):
        os.mkdir(dir)
        try:
            for recordfile in os.listdir('03_separated'):
                if recordfile.endswith('.tfrecord'):
                    prefix = os.path.splitext(recordfile)[0]
                    cmd = [
                        'python',
                        '-m',
                        'scripts.tfrecord_to_lmdb',
                        f'03_separated/{recordfile}',
                        f'{tmp_dir}/{prefix}.db'
                    ]
                    subprocess.check_call(cmd)
                    os.unlink(f'{tmp_dir}/{prefix}.db-lock')
                    for dbfile in os.listdir(tmp_dir):
                        if dbfile.startswith(prefix):
                            shutil.move(os.path.join(tmp_dir, dbfile), os.path.join(dir, dbfile))
        except subprocess.CalledProcessError:
            die("Command failed: " + " ".join(cmd))

    dir = "final"
    os.mkdir(dir)
    if os.path.isdir(dir):
        for filename in os.listdir("03_separated"):
            src_file = os.path.join("03_separated", filename)
            dest_file = os.path.join(dir, filename)
            os.link(src_file, dest_file)

        for filename in os.listdir("04_db"):
            src_file = os.path.join("04_db", filename)
            dest_file = os.path.join(dir, filename)
            os.link(src_file, dest_file)

        # Add keys, song names and styles to the metadata.
        with gzip.open("02_chopped/data_meta.json.gz", "rt") as f:
            data = json.load(f)
            data_dict = {}
            key_len = len(str(len(data) - 1))
            for i, item in enumerate(data):
                item["song_name"], item["style"], _ = item["filename"].rsplit(".", maxsplit=2)
                key = str(i).zfill(key_len)
                data_dict[key] = item
            with gzip.open(f"{dir}/meta.json.gz", "wt") as f:
                json.dump(data_dict, f, separators=(",", ":"))

        # Shuffle the data.
        os.mkdir(f"{dir}/shuf")
        with open(f"{dir}/shuf/key_map", "w") as f:
            keys = subprocess.check_output(["python", "-m", "scripts.list_lmdb_keys", f"{dir}/all.db"])
            keys = keys.decode().splitlines()
            random.shuffle(keys)
            for k1, k2 in zip(keys, sorted(keys)):
                f.write(f"{k1}\t{k2}\n")
        for instr in ["Bass", "Piano", "Guitar", "Strings", "Drums", "all", "all_except_drums"]:
            log_progress(instr)
            subprocess.run(["python", "-m", "scripts.permute_lmdb", f"{dir}/{instr}.db", f"{dir}/shuf/{instr}.db", f"{dir}/shuf/key_map"], check=True)
        subprocess.run(["python", "-m", "scripts.permute_json_map", f"{dir}/meta.json.gz", f"{dir}/shuf/meta.json.gz", f"{dir}/shuf/key_map"], check=True)

        log("Done.")
    else:
        raise ValueError(f"Directory '{dir}' could not be created.")

finally:
    cleanup()
