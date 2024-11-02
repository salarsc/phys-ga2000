import numpy as np
import matplotlib.pyplot as plt

# Load audio signal data from the provided text files

piano_waveform = np.loadtxt('piano.txt')
trumpet_waveform = np.loadtxt('trumpet.txt')

# Define the sampling frequency
sample_rate = 44100  # Standard audio sample rate in Hz

# Step 1: Visualize the raw audio waveforms for Piano and Trumpet
plt.figure(figsize=(10, 4))
plt.plot(piano_waveform, label="Piano Sound", color="navy", alpha=0.7)
plt.plot(trumpet_waveform, label="Trumpet Sound", color="coral", alpha=0.7)
plt.title("Raw Audio Waveforms: Piano and Trumpet")
plt.xlabel("Sample Number")
plt.ylabel("Sound Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("q2-1.png")
plt.show()

# Step 2: Perform FFT and display the amplitude spectrum of the first 10,000 frequencies
# Use Fast Fourier Transform (FFT) to compute frequency domain representations
piano_fft = np.fft.fft(piano_waveform)
trumpet_fft = np.fft.fft(trumpet_waveform)

# Get the magnitude of the first 10,000 frequency components
piano_amplitude_spectrum = np.abs(piano_fft[:10000])
trumpet_amplitude_spectrum = np.abs(trumpet_fft[:10000])

plt.figure(figsize=(10, 4))
plt.plot(piano_amplitude_spectrum, label="Piano Spectrum", color="navy", alpha=0.7)
plt.plot(trumpet_amplitude_spectrum, label="Trumpet Spectrum", color="coral", alpha=0.7)
plt.title("Amplitude Spectrum Comparison: Piano and Trumpet")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude Magnitude")
plt.legend()
plt.grid(True)
plt.savefig("q2-2.png")
plt.show()

# Step 3: Calculate and output the primary frequency (fundamental frequency) for both waveforms
# Generate the frequency axis for each signal based on sampling rate
piano_frequencies = np.fft.fftfreq(len(piano_waveform), d=1/sample_rate)
trumpet_frequencies = np.fft.fftfreq(len(trumpet_waveform), d=1/sample_rate)

# Find the main frequency component (fundamental) by identifying the highest peak
primary_freq_piano = piano_frequencies[np.argmax(piano_amplitude_spectrum[:10000])]
primary_freq_trumpet = trumpet_frequencies[np.argmax(trumpet_amplitude_spectrum[:10000])]

# Display the primary frequency for both instruments
print("Primary Frequency of Piano:", primary_freq_piano, "Hz")
print("Primary Frequency of Trumpet:", primary_freq_trumpet, "Hz")

# Step 4: Determine the musical notes played
# Reference frequency for middle C (C4)
frequency_middle_C = 261.63

# Calculate the number of semitones from middle C for each frequency
semitones_piano = 12 * np.log2(primary_freq_piano / frequency_middle_C)
semitones_trumpet = 12 * np.log2(primary_freq_trumpet / frequency_middle_C)

# Find the closest musical note by rounding to the nearest semitone
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
note_piano = note_names[round(semitones_piano) % 12]
note_trumpet = note_names[round(semitones_trumpet) % 12]

# Display the musical notes corresponding to the primary frequencies
print("Musical Note for Piano:", note_piano)
print("Musical Note for Trumpet:", note_trumpet)
