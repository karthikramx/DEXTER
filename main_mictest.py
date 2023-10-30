# whisper push to talk (spacebar)
# Command to run: sudo python whisper_push_to_talk.py --device_index 2 --rate 48000 --channel 1 --model 'tiny.en' 
# If you are on Mac: list the audio devices -> system_profiler SPAudioDataType 
import os
from dotenv import load_dotenv
from deepgram import Deepgram
import asyncio, json
import sounddevice as sd
from scipy.io import wavfile
import numpy as np
from datetime import datetime as time
import argparse
import keyboard
import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()
parser.add_argument('--devices', default='False', type=str, help='print all available devices id')
parser.add_argument('--model', type=str, choices=['tiny','tiny.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], default='small', help='model to be use for generating audio transcribe')
parser.add_argument('--device_index', default= 2, type=int, help='the id of the device ')
parser.add_argument('--channel', default= 1, type=int, help='number of channels for the device')
parser.add_argument('--rate', default= 44100, type=int, help="polling rate of the output device")
args = parser.parse_args()


load_dotenv()

# Your Deepgram API Key
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = Deepgram(DEEPGRAM_API_KEY)


frames = []

def callback(indata,a,b,c):
        global frames
        frames.append(indata.copy())

async def get_transcription(filename):
    with open(filename, 'rb') as audio:
        source = { 'buffer': audio, 'mimetype': 'audio/wav' }
        transcription_options = { 'punctuate': True }
        response = await deepgram.transcription.prerecorded(source, transcription_options)
        print(response["results"]["channels"][0]["alternatives"][0]["transcript"])


def main():
    global frames
    rate: int=args.rate
    channel: int=args.channel
    device_index: int=args.device_index

    sd.default.device[0] = device_index
    sd.default.dtype[0] = np.float32

    index = 0

    while True:
        try:
            print("Start Recording...")
            while True:
                if keyboard.is_pressed(' '):
                    break
            if keyboard.is_pressed(' '):
                while keyboard.is_pressed(' '):
                    print("recoding audio")
                    stream = sd.InputStream(callback=callback, channels=channel, samplerate=rate)
                    stream.start()
                    while keyboard.is_pressed(' '):
                        print("recording...",end="\r")
                    break

                # Stop recording
                stream.stop()
                stream.close()
                recording = np.concatenate(frames)
                
                frames = []

                sd.wait()
                wavfile.write(f"./audio/audio{index}.wav", rate=rate, data=recording)
                asyncio.run(get_transcription(f"./audio/audio{index}.wav"))
                
                index += 1
        except Exception as e:
            print("Exception:",e)
            pass


if __name__ == '__main__':
    main()






