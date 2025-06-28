"""
Audio Handler
Simple VAD-based handler for real-time transcription with rolling buffer (max 20s)
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import time
from typing import Optional, Callable, Tuple
import logging
import webrtcvad

# Set up logging
logger = logging.getLogger(__name__)


class AudioHandler:
    """Simple VAD-based audio handler with rolling buffer (max 20s)"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio handler with VAD.

        Args:
            sample_rate: Sample rate for audio capture (16000 Hz is required for VAD and Whisper).
            channels: Number of audio channels (must be 1 for VAD).
        """
        if sample_rate != 16000:
            raise ValueError("Sample rate must be 16000 Hz for VAD.")
        if channels != 1:
            raise ValueError("Channels must be 1 (mono) for VAD.")

        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.paused = False
        self.vad = webrtcvad.Vad(2)
        self.frame_ms = 30
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.silence_timeout_ms = 500
        self.min_speech_ms = 200
        self.audio_queue = queue.Queue()
        self.segment_queue = queue.Queue()
        self.producer_thread = None
        self.consumer_thread = None
        self.speech_buffer = []
        self.segment_callback = None
        self.level_callback = None
        self.max_segment_seconds = 20
        self.simple_audio_buffer = []

    def list_devices(self) -> list:
        """List all available audio devices"""
        devices = sd.query_devices()
        device_list = []
        for i, device in enumerate(devices):
            device_list.append({
                'index': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate'],
                'is_default': i == sd.default.device[0]
            })
        return device_list
        
    def set_input_device(self, device_index: Optional[int] = None):
        """Set the input device for recording"""
        if device_index is not None:
            sd.default.device[0] = device_index
            
    def start_recording(self, segment_callback: Optional[Callable] = None, level_callback: Optional[Callable] = None):
        """
        Start recording audio with VAD pipeline.
        """
        if self.recording:
            logger.warning("Already recording")
            return

        self.recording = True
        self.paused = False
        self.segment_callback = segment_callback
        self.level_callback = level_callback
        self.speech_buffer = []

        self.producer_thread = threading.Thread(target=self._audio_producer, daemon=True)
        self.producer_thread.start()
        
        self.consumer_thread = threading.Thread(target=self._audio_consumer, daemon=True)
        self.consumer_thread.start()

        logger.info("✓ Recording started with simple VAD pipeline")

    def _audio_producer(self):
        """Producer thread: captures audio and enqueues raw frames."""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            if not self.recording:
                return

            self.audio_queue.put(indata.copy())
            if self.level_callback:
                level = int(min(100, float(np.abs(indata).max()) / 32767.0 * 100))
                self.level_callback(level)

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                dtype='int16',  # VAD requires 16-bit PCM
                blocksize=self.frame_samples
            )
            self.stream.start()
            while self.recording:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"✗ Error in audio producer: {e}")
            self.recording = False
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

    def _audio_consumer(self):
        """Consumer thread: processes raw audio with VAD to create speech segments."""
        silence_frames = 0
        num_silence_frames = self.silence_timeout_ms // self.frame_ms
        
        while self.recording:
            try:
                frame = self.audio_queue.get(timeout=1.0)
                if self.paused:
                    continue
                
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                if is_speech:
                    self.speech_buffer.append(frame)
                    silence_frames = 0
                elif self.speech_buffer:
                    silence_frames += 1
                    if silence_frames > num_silence_frames:
                        self._emit_segment()
                        
            except queue.Empty:
                if self.speech_buffer:
                    self._emit_segment() # Process buffer on timeout
                continue
            except Exception as e:
                logger.error(f"✗ Error in audio consumer: {e}")
                break
        
        if self.speech_buffer:
            self._emit_segment() # Process any remaining audio

    def _emit_segment(self):
        """Process the buffered speech frames into a segment."""
        if not self.speech_buffer:
            return

        segment = np.concatenate(self.speech_buffer)
        self.speech_buffer.clear()
        
        min_samples = int(self.sample_rate * self.min_speech_ms / 1000)
        if len(segment) < min_samples:
            return # Ignore segments that are too short

        # --- HARD CAP: only keep first 5s ---
        max_samples = self.sample_rate * 5
        if len(segment) > max_samples:
            segment = segment[:max_samples]

        # Convert to float32 for Whisper
        segment = segment.astype(np.float32) / 32768.0
        
        self.segment_queue.put(segment)
        if self.segment_callback:
            try:
                self.segment_callback(segment, self.sample_rate)
            except Exception as e:
                logger.error(f"Error in segment_callback: {e}")

    def pause_recording(self):
        """Pause audio processing without stopping capture."""
        self.paused = True
        logger.info("Recording paused")

    def resume_recording(self):
        """Resume audio processing."""
        self.paused = False
        logger.info("Recording resumed")

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if not self.recording:
            return
            
        self.recording = False
        logger.info("Stopping recording...")
        
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=2.0)
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=2.0)
            
        logger.info("✓ Recording stopped")

    def get_next_segment(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next audio segment from the queue."""
        try:
            return self.segment_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queues(self):
        """Clear all audio queues."""
        for q in [self.audio_queue, self.segment_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self.speech_buffer = []

    def save_recording(self, filename: str, audio_data: Optional[np.ndarray] = None) -> bool:
        """
        Save recorded audio to file
        
        Args:
            filename: Output filename
            audio_data: Optional audio data to save. If None, uses last recording.
            
        Returns:
            True if saved successfully, False otherwise
        """
        if audio_data is None and self.speech_buffer:
            audio_data = np.concatenate(self.speech_buffer).flatten()
        elif audio_data is None:
            logger.warning("No audio data to save")
            return False
            
        try:
            sf.write(filename, audio_data, self.sample_rate)
            logger.info(f"✓ Audio saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"✗ Error saving audio: {e}")
            return False
            
    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """
        Play audio data
        
        Args:
            audio_data: Audio data to play
            sample_rate: Sample rate (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until playback is finished
        except Exception as e:
            logger.error(f"✗ Error playing audio: {e}")
            
    def get_recording_duration(self) -> float:
        """Get the duration of the current recording in seconds"""
        if not self.speech_buffer:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self.speech_buffer)
        return total_samples / self.sample_rate
        
    def get_statistics(self) -> dict:
        """Get recording statistics"""
        return {
            "recording": self.recording,
            "paused": self.paused,
            "total_samples": len(self.speech_buffer),
            "duration": self.get_recording_duration(),
            "queue_size": self.segment_queue.qsize()
        }

    # Simple continuous recording methods (no VAD)
    def start_simple_recording(self):
        """Start simple continuous recording without VAD."""
        if self.recording:
            logger.warning("Already recording")
            return

        self.recording = True
        self.simple_audio_buffer = []
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            if not self.recording:
                return
            # Store all audio data
            self.simple_audio_buffer.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                dtype='int16',
                blocksize=1024
            )
            self.stream.start()
            logger.info("✓ Simple recording started")
        except Exception as e:
            logger.error(f"✗ Error starting simple recording: {e}")
            self.recording = False

    def stop_simple_recording(self) -> Tuple[Optional[np.ndarray], int]:
        """Stop simple recording and return the audio data."""
        if not self.recording:
            return None, self.sample_rate
            
        self.recording = False
        logger.info("Stopping simple recording...")
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.simple_audio_buffer:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.simple_audio_buffer)
            # Convert to float32 for Whisper
            audio_data = audio_data.astype(np.float32) / 32768.0
            logger.info("✓ Simple recording stopped")
            return audio_data, self.sample_rate
        else:
            logger.warning("No audio data recorded")
            return None, self.sample_rate 