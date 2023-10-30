import asyncio
import os
from dotenv import load_dotenv
from deepgram import Deepgram

load_dotenv()
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
deepgram = Deepgram(DEEPGRAM_API_KEY)


async def main():
    with open(f'./audio/audio0.wav', 'rb') as audio:
        source = { 'buffer': audio, 'mimetype': 'audio/wav' }
        transcription_options = { 'punctuate': True }
        response = await deepgram.transcription.prerecorded(source, transcription_options)
        print(response["results"]["channels"][0]["alternatives"][0]["transcript"])

if __name__ == '__main__':
    asyncio.run(main())

