import numpy as np
import scipy.io.wavfile as wav
import sys

def check_bandpass(file_path):
    print(f"🔍 Analyzing: {file_path}")
    
    try:
        # Load the file
        sr, data = wav.read(file_path)
        
        # If stereo, take one channel
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Normalize
        data = data / np.max(np.abs(data))
        
        # Perform FFT (Fast Fourier Transform) to get frequency content
        fft_out = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/sr)
        
        # Take magnitude and ignore negative frequencies
        magnitude = np.abs(fft_out)
        magnitude = magnitude[:len(magnitude)//2]
        freqs = freqs[:len(freqs)//2]
        
        # Define Bands
        # 1. RUMBLE (< 100 Hz) - Hand noise, wind, thuds
        low_mask = (freqs < 100)
        low_energy = np.sum(magnitude[low_mask])
        
        # 2. MEDICAL ZONE (100 - 2000 Hz) - Wheezes and Crackles
        mid_mask = (freqs >= 100) & (freqs <= 2000)
        mid_energy = np.sum(magnitude[mid_mask])
        
        # 3. STATIC (> 2000 Hz) - Electronic hiss
        high_mask = (freqs > 2000)
        high_energy = np.sum(magnitude[high_mask])
        
        total_energy = low_energy + mid_energy + high_energy
        
        # Percentages
        p_low = (low_energy / total_energy) * 100
        p_mid = (mid_energy / total_energy) * 100
        p_high = (high_energy / total_energy) * 100
        
        print("-" * 30)
        print(f"📉 Low Bass (<100Hz):   {p_low:.1f}%  {'🔴 (Too High)' if p_low > 20 else '✅ (Clean)'}")
        print(f"🫁 Medical (100-2k):    {p_mid:.1f}%")
        print(f"📈 High Static (>2k):   {p_high:.1f}% {'🔴 (Too High)' if p_high > 20 else '✅ (Clean)'}")
        print("-" * 30)

        if p_mid > 80:
            print("✅ VERDICT: Bandpass Filter is WORKING perfectly.")
        elif p_low > 40:
            print("❌ VERDICT: Heavy Rumble Detected (Filter missing or weak).")
        else:
            print("⚠️ VERDICT: Audio is noisy/unfiltered.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_filter.py <filename.wav>")
    else:
        check_bandpass(sys.argv[1])