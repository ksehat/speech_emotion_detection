import pyaudio
import wave
import time

# The following values are typical for most microphones
CHUNK = 1024  # Record in chunks of 1024 samples
SAMPLE_FORMAT = pyaudio.paInt16  # 16 bits per sample
CHANNELS = 2
FS = 44100  # Record at 44100 samples per second
SECONDS = 3
FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=FS,
                frames_per_buffer=CHUNK,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(FS / CHUNK * SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
wf.setframerate(FS)
wf.writeframes(b''.join(frames))
wf.close()

# Now you can send the recorded audio (saved in 'output.wav') to your model
