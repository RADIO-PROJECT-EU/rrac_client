import os
import struct
import wave

def fix_wav_header(filename):
    try:
        with open(filename, 'r+b') as f:
            size_in_bytes = os.path.getsize(filename)

            # Size of file.
            f.seek(4)
            for c in struct.pack('<I', (size_in_bytes-8)):
                f.write(c)

            # Size of data section.
            f.seek(40)
            for c in struct.pack('<I', (size_in_bytes-44)):
                f.write(c)
    except IOError:
        logging.error('I/O error in accessing {}'.format(filename))

def wav_duration(filename):
    w = wave.open(filename, 'r')

    nframes = w.getnframes()
    rate = float(w.getframerate())
    w.close()
    return nframes / rate
    
def split_recording(filename, output_dir, count_offset=0, duration=1000, overlap=0.0, output_file_format='sample{0:04d}.wav'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = count_offset
    in_wavefile = wave.open(filename, 'rb')
    samplerate = in_wavefile.getframerate()
    channels = in_wavefile.getnchannels()
    samplewidth = in_wavefile.getsampwidth()

    out_block_nframes = int(duration * samplerate / 1000.0)
    out_block_nbytes = out_block_nframes * channels * samplewidth
    noverlap_frames = int(out_block_nframes * overlap)

    while True:
        frames = in_wavefile.readframes(out_block_nframes)
        if not len(frames) == out_block_nbytes:
            break

        out_filename = os.path.join(output_dir, output_file_format.format(count))
        out_wavefile = wave.open(out_filename, 'w')
        out_wavefile.setframerate(samplerate)
        out_wavefile.setnchannels(channels)
        out_wavefile.setsampwidth(samplewidth)
        out_wavefile.writeframes(frames)
        out_wavefile.close()

        count += 1
        in_wavefile.setpos(in_wavefile.tell() - noverlap_frames)

    return count
