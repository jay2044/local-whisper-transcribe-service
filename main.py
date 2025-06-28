#!/usr/bin/env python3
"""
Simple demonstration of the PyQt OpenAI Whisper Local Service
"""
import sys
import numpy as np
from src.core.whisper_model import WhisperModel
from src.core.audio_handler import AudioHandler
import sounddevice as sd


def test_whisper_model():
    """Test Whisper model loading and basic functionality"""
    print("\n" + "=" * 60)
    print("Testing Whisper Model")
    print("=" * 60)
    
    # Initialize model
    model = WhisperModel(model_size="base")
    
    # Load the model
    print("\nLoading Whisper model...")
    model.load_model()
    
    # Get model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Show available languages
    languages = model.get_available_languages()
    print(f"\nAvailable languages: {len(languages)}")
    print(f"Examples: {', '.join(languages[:10])}...")
    

def test_audio_handler():
    """Test audio handler functionality"""
    print("\n" + "=" * 60)
    print("Testing Audio Handler")
    print("=" * 60)
    
    # Initialize audio handler
    audio = AudioHandler()
    
    # List audio devices
    devices = audio.list_devices()
    print(f"\nFound {len(devices)} audio devices:")
    for device in devices:
        default = " (DEFAULT)" if device['is_default'] else ""
        print(f"  [{device['index']}] {device['name']}{default}")
        print(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']}")
    

def demo_recording():
    """Demonstrate recording capability"""
    print("\n" + "=" * 60)
    print("Recording Demo")
    print("=" * 60)
    
    audio = AudioHandler()
    
    print("\nPress Enter to start recording...")
    input()
    
    # Start simple continuous recording
    audio.start_simple_recording()
    
    print("Recording... Press Enter to stop.")
    input()
    
    # Stop recording and get audio data
    audio_data, sample_rate = audio.stop_simple_recording()
    
    if audio_data is not None:
        duration = len(audio_data) / sample_rate
        print(f"\nRecorded {duration:.2f} seconds of audio")
        
        # Save the recording
        filename = "test_recording.wav"
        if audio.save_recording(filename, audio_data):
            print(f"Recording saved to {filename}")
            
            # Transcribe the recording
            print("\nTranscribing the recording...")
            model = WhisperModel(model_size="base")
            model.load_model()
            
            result = model.transcribe_audio(filename)
            print(f"\nTranscription: {result['text']}")
    else:
        print("\nNo audio recorded.")


def demo_live_transcription():
    """Demonstrate live transcription capability with 5-second clips"""
    print("\n" + "=" * 60)
    print("Live Transcription Demo (5-second clips)")
    print("=" * 60)
    
    import threading
    import queue
    import time
    
    # Initialize components
    audio = AudioHandler()
    model = WhisperModel(model_size="base")
    
    print("\nLoading Whisper model...")
    if not model.load_model():
        print("Failed to load model!")
        return
    
    print("✓ Model loaded successfully")
    
    # Queue for audio clips
    audio_queue = queue.Queue()
    transcription_queue = queue.Queue()
    running = True
    
    def audio_recorder():
        """Record audio in 5-second clips"""
        clip_duration = 5  # seconds
        samples_per_clip = int(audio.sample_rate * clip_duration)
        current_clip = []
        silence_threshold = 0.003  # Same threshold as sentence transcription
        
        def audio_callback(indata, frames, time, status):
            nonlocal current_clip
            if status:
                print(f"\r⚠️  Audio status: {status}", end="", flush=True)
            if not running:
                return
            
            # Add audio data to current clip
            current_clip.extend(indata.flatten())
            
            # When we have enough samples for a clip, queue it
            while len(current_clip) >= samples_per_clip:
                # Extract 5-second clip
                clip_data = np.array(current_clip[:samples_per_clip], dtype=np.float32) / 32768.0
                current_clip = current_clip[samples_per_clip:]
                
                # Only queue if clip contains real speech (not just silence)
                clip_rms = np.sqrt(np.mean(clip_data ** 2))
                if clip_rms > silence_threshold:
                    audio_queue.put(clip_data)
                    print(f"\r🎵 Queued clip #{audio_queue.qsize()} (RMS={clip_rms:.4f})", end="", flush=True)
                else:
                    print(f"\r🔇 Skipped silent clip (RMS={clip_rms:.4f})", end="", flush=True)
        
        try:
            stream = sd.InputStream(
                samplerate=audio.sample_rate,
                channels=audio.channels,
                callback=audio_callback,
                dtype='int16',
                blocksize=1024
            )
            stream.start()
            print(f"\n🎙️  Recording started! (5-second clips)")
            
            while running:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
        finally:
            if 'stream' in locals():
                stream.stop()
                stream.close()
    
    def transcription_worker():
        """Transcribe audio clips from queue"""
        while running or not audio_queue.empty():
            try:
                # Get clip from queue (timeout to check if still running)
                clip_data = audio_queue.get(timeout=1.0)
                
                # Transcribe the clip
                result = model.transcribe_array(clip_data, audio.sample_rate)
                text = result['text'].strip()
                
                if text:
                    print(text)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
    
    print("\nPress Enter to start live transcription...")
    print("(Press Enter again to stop)")
    input()
    
    # Start recording and transcription threads
    recorder_thread = threading.Thread(target=audio_recorder, daemon=True)
    transcriber_thread = threading.Thread(target=transcription_worker, daemon=True)
    
    recorder_thread.start()
    transcriber_thread.start()
    
    print("Press Enter to stop...")
    input()
    
    # Stop recording
    running = False
    print("\n🛑 Stopping...")
    
    # Wait for threads to finish
    recorder_thread.join(timeout=2.0)
    transcriber_thread.join(timeout=2.0)
    
    # Collect all transcriptions
    transcriptions = []
    while not transcription_queue.empty():
        transcriptions.append(transcription_queue.get())
    
    # Show final result
    if transcriptions:
        final_text = " ".join(transcriptions)
        print(f"\n📝 Complete transcript:")
        print(f"'{final_text}'")
        print(f"\n📊 Processed {len(transcriptions)} clips")
    else:
        print("\n📝 No speech detected during recording.")


def demo_sentence_transcription():
    """Transcribe one sentence at a time (no VAD, silence-based)"""
    print("\n" + "=" * 60)
    print("Sentence-by-Sentence Transcription Demo")
    print("=" * 60)
    
    import threading
    import queue
    import time
    
    # Parameters
    sample_rate = 16000
    channels = 1
    silence_threshold = 0.003  # Lowered threshold for more sensitivity
    min_silence_duration = 0.8  # seconds of silence to end a sentence
    min_sentence_duration = 0.7  # minimum length of a sentence (seconds)
    chunk_size = 1024  # samples per callback
    
    # Initialize model
    model = WhisperModel(model_size="base")
    print("\nLoading Whisper model...")
    if not model.load_model():
        print("Failed to load model!")
        return
    print("✓ Model loaded successfully")
    
    running = True
    audio_buffer = []
    last_audio_time = time.time()
    last_non_silent = time.time()
    sentence_queue = queue.Queue()
    
    def audio_callback(indata, frames, t, status):
        nonlocal audio_buffer, last_non_silent
        if status:
            print(f"\r⚠️  Audio status: {status}", end="", flush=True)
        if not running:
            return
        # Convert to float32
        samples = indata.flatten().astype(np.float32) / 32768.0
        audio_buffer.extend(samples)
        # Compute RMS energy
        rms = np.sqrt(np.mean(samples ** 2))
        now = time.time()
        if rms > silence_threshold:
            last_non_silent = now
        # If enough silence and enough audio, treat as sentence
        if (now - last_non_silent > min_silence_duration and
            len(audio_buffer) > int(min_sentence_duration * sample_rate)):
            # Only queue if buffer contains real speech (not just silence)
            buffer_rms = np.sqrt(np.mean(np.array(audio_buffer) ** 2))
            if buffer_rms > silence_threshold:
                sentence = np.array(audio_buffer, dtype=np.float32)
                sentence_queue.put(sentence)
                print(f"\n📝 Detected end of sentence. Queued for transcription. (RMS={buffer_rms:.4f})")
            else:
                print(f"\n🔇 (ignored silence, RMS={buffer_rms:.4f})")
            audio_buffer = []
    
    def transcription_worker():
        while running or not sentence_queue.empty():
            try:
                sentence = sentence_queue.get(timeout=1.0)
                result = model.transcribe_array(sentence, sample_rate)
                text = result['text'].strip()
                if text:
                    print(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
    
    print("\nPress Enter to start sentence-by-sentence transcription...")
    print("(Speak a sentence, pause, and see the result. Press Enter again to stop.)")
    input()
    
    # Start audio stream and transcription thread
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        callback=audio_callback,
        dtype='int16',
        blocksize=chunk_size
    )
    transcriber_thread = threading.Thread(target=transcription_worker, daemon=True)
    
    stream.start()
    transcriber_thread.start()
    print("\n🎙️  Speak a sentence, pause, and see the result. Press Enter to stop...")
    input()
    running = False
    stream.stop()
    stream.close()
    transcriber_thread.join(timeout=2.0)
    print("\n🛑 Stopped.")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("PyQt OpenAI Whisper Local Service - Demo")
    print("=" * 60)
    
    while True:
        print("\nSelect an option:")
        print("1. Test Whisper Model")
        print("2. Test Audio Handler") 
        print("3. Recording Demo (Record and Transcribe)")
        print("4. Live Transcription Demo")
        print("5. Sentence-by-Sentence Transcription Demo")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            test_whisper_model()
        elif choice == "2":
            test_audio_handler()
        elif choice == "3":
            demo_recording()
        elif choice == "4":
            demo_live_transcription()
        elif choice == "5":
            demo_sentence_transcription()
        elif choice == "6":
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main() 