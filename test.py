import os
import time
import multiprocessing
import warnings
from scipy.io import wavfile
warnings.filterwarnings("ignore")

audio_queue = multiprocessing.Queue()
audio_segments_filename_queue = multiprocessing.Queue()
text_queue = multiprocessing.Queue()

[os.remove('./audio/'+ filename) for filename in sorted(os.listdir('./audio')) if filename.startswith('temp') and filename.endswith('.wav') ]


############################################################
###################### CORE FUNCS ##########################
############################################################


def queue_audio(queue):
    "Simply adds audio chunks to the audio queue"

    import pyaudio

    # Audio Configurations / vars
    i = 0
    start_set = False
    SAMPLING_RATE = 16000
    SAMPLE_WIDTH = 2
    CHANNELS = 1
    CHUNK_SIZE = 1024 * 5
    RATE = 48000

    # Initialize PyAudio / Mic
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(SAMPLE_WIDTH),
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE,
                        input_device_index=2)

    print("Start talking...")
    while True:
        # Collect audio chunk from the microphone
        chunk = stream.read(CHUNK_SIZE,exception_on_overflow=False)
        queue.put(chunk)


def voice_activity_audio_segmentor(audio_q, audio_segment_filename_q):

    import resampy
    import numpy as np
    import wave
    import torch
    from typing import List


    audio_data = b''
    audio_chunk_index = 1
    start = 0
    end = 0
    last_end = 0
    prev_end = 0
    audio_crop_index = 0
    audio_segment_cont = False

    # Load Silero VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=False,
                            onnx=False)
    
    # Getting necessary utilities
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
     
    # audio resampler
    def resample_audio_data(chunk:bytes) -> bytes:
        chunk = np.frombuffer(chunk, dtype=np.int16)
        chunk = resampy.resample(chunk, 44100, 16000)
        chunk = chunk.astype(np.int16).tobytes()
        return chunk

    # storing raw audio as wav file
    def store_to_wav(filename: str, audio_data: bytes) -> None:
        with wave.open(filename, 'wb') as file:
                                    file.setnchannels(1)                        
                                    file.setsampwidth(2)                       
                                    file.setframerate(16000)                    
                                    file.writeframes(audio_data) 

        ## using VADIterator class
    
## using VADIterator class
    def get_time_stamps(filename: str,vad_model):
        vad_iterator = VADIterator(vad_model, threshold = 0.2,
                                    sampling_rate = 16000,
                                    min_silence_duration_ms = 50,
                                    speech_pad_ms = 300)
        wav = read_audio(filename, sampling_rate=16000)
        speech_time_stamps = []
        window_size_samples = 512 # number of samples in a single audio chunk
        for i in range(0, len(wav), window_size_samples):
            try:
                speech_dict = vad_iterator(wav[i: i+ window_size_samples], return_seconds=True)
                if speech_dict:
                    speech_time_stamps.append(speech_dict)
            except:
                pass
        vad_iterator.reset_states() # reset model states after each audio
        return speech_time_stamps,wav
    
    def collect_chunksx(tss: List[dict],wav: torch.Tensor):
        chunks = []
        start_vals = []
        end_vals = []
        for ele in tss:
            if 'start' in ele: start_vals.append(ele['start'])
            if 'end' in ele: end_vals.append(ele['end'])
        zipper = zip(start_vals,end_vals)

        try:
            for start,end in zipper:
                chunks.append(wav[int(start*16000):int(end*16000)])
        except:
            pass
        if chunks == []:
            chunks.append(wav)
        return torch.cat(chunks)
    
    def crop_current_incremental_audio(wav_history,start,stop,audio_crop_index):
         "Crop current incremental audio"
         # print("Audio crop at:",crop_at)
         audio_data = wav_history[int(start*16000):int(stop*16000)]

         filename = f'./audio/tempx{str(audio_crop_index).zfill(3)}.wav'
         audio_crop_index = audio_crop_index + 1
         save_audio(path=filename,tensor= audio_data)
         return filename, audio_crop_index
    
    def get_last_end_timestamp(time_stamp_data) -> int:
        if time_stamp_data == []:
            return None
        last_end_index = None
        for i in range(len(time_stamp_data) - 1, -1, -1):
            if 'end' in time_stamp_data[i]:
                last_end_index = i
                break
        if last_end_index is not None:
             return time_stamp_data[last_end_index]['end']
        else:
             return last_end_index
    
    while True:
        while audio_q:
            audio_chunk = audio_q.get()
            audio_chunk = resample_audio_data(audio_chunk)
            audio_data += audio_chunk
            if len(audio_data) > 64000*audio_chunk_index:
                ### Stable #### -> Stores Audio as a wav file
                ### Storing the 2-second incremental chunk
                audio_save = audio_data[0:64000*audio_chunk_index]

                filename = f'./audio/temp{str(audio_chunk_index).zfill(3)}.wav'
                store_to_wav(filename=filename,audio_data=audio_save)

                wav_main = read_audio(filename, sampling_rate=16000)
                prev_end = end

                look_forward_audio = wav_main[int(prev_end*16000):]
                filename_look_forward = f'./audio/tempf{str(audio_chunk_index).zfill(3)}.wav'
                save_audio(filename_look_forward,look_forward_audio)


                speech_time_stamps,wav_look_forward = get_time_stamps(filename_look_forward,model)
                end = get_last_end_timestamp(speech_time_stamps)

                if end is not None and end != prev_end:
                    filename, audio_crop_index = crop_current_incremental_audio(wav_history=wav_look_forward,start=0, stop=end, audio_crop_index=audio_crop_index)
                    audio_segment_filename_q.put(filename)
                if end is None:
                    end = prev_end
                else:
                    end = end + prev_end

                audio_chunk_index = audio_chunk_index + 1   
        

def getTranscription(audio_segment_filename_q,text_q):
    pass


if __name__ == "__main__":
    # starting audio process...
    queue_audio_process = multiprocessing.Process(target=queue_audio, args=(audio_queue,))
    queue_audio_process.start()

    # starting voice activity detection queue
    vad_process = multiprocessing.Process(target=voice_activity_audio_segmentor, args=(audio_queue,audio_segments_filename_queue))
    vad_process.start()
    
    previous_transcription = None
    try:
        # Wait for the keyboard interrupt (Ctrl+C)
        while True:
            while text_queue:
                 transcription = text_queue.get()
                 if previous_transcription != transcription:
                    print(transcription,end="\n")
                 previous_transcription = transcription
                 time.sleep(0.10)
    except KeyboardInterrupt:
        print("Stopping processes...")
        queue_audio_process.terminate()
        #transcription_process.terminate()
        vad_process.terminate()


# Next step are to add the file names in a audio_segment_queue and have them transcribed over in a different process


