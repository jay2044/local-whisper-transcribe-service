"""
Main Window for Local Whisper Transcriber
PyQt6 interface with live and file transcription modes
"""
import sys
import os
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import queue
import tempfile
import subprocess

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QComboBox, QLabel, QProgressBar, QFileDialog,
    QGroupBox, QGridLayout, QMessageBox, QStatusBar, QSplitter, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QObject
from PyQt6.QtGui import QFont, QIcon

from ..core.whisper_model import WhisperModel
from ..core.audio_handler import AudioHandler
import soundfile as sf
import numpy as np
import sounddevice as sd
import scipy.signal
import soundcard as sc
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class SpeakerDiarizer:
    """Enhanced speaker diarization with multiple clustering algorithms and speaker naming"""
    
    def __init__(self):
        self.encoder = VoiceEncoder()
        self.speaker_names = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
            "Iris", "Jack", "Kate", "Liam", "Maya", "Noah", "Olivia", "Paul"
        ]
        self.clustering_methods = {
            'agglomerative': AgglomerativeClustering,
            'dbscan': DBSCAN,
            'spectral': SpectralClustering
        }
        
    def extract_embeddings(self, audio_segments: List[np.ndarray]) -> List[np.ndarray]:
        """Extract voice embeddings from audio segments"""
        embeddings = []
        for audio in audio_segments:
            try:
                # Ensure audio is in the right format
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()
                
                # Extract embedding
                emb = self.encoder.embed_utterance(audio)
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Could not extract embedding for segment: {e}")
                # Use zero embedding as fallback
                embeddings.append(np.zeros(256))
        
        return embeddings
    
    def cluster_speakers(self, embeddings: List[np.ndarray], method: str = 'auto') -> List[int]:
        """Cluster speakers using the best available method"""
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        embeddings_array = np.array(embeddings)
        
        # Try different clustering methods and select the best one
        best_labels = None
        best_score = -1
        
        methods_to_try = ['agglomerative', 'dbscan', 'spectral'] if method == 'auto' else [method]
        
        for cluster_method in methods_to_try:
            try:
                if cluster_method == 'agglomerative':
                    # Try different distance thresholds
                    thresholds = [0.6, 0.65, 0.7, 0.75]
                    for threshold in thresholds:
                        clustering = AgglomerativeClustering(
                            n_clusters=None, 
                            distance_threshold=threshold, 
                            linkage="average"
                        )
                        labels = clustering.fit_predict(embeddings_array)
                        
                        if len(set(labels)) > 1:  # More than one cluster
                            try:
                                score = silhouette_score(embeddings_array, labels)
                                if score > best_score:
                                    best_score = score
                                    best_labels = labels
                            except:
                                # If silhouette score fails, use this clustering anyway
                                if best_labels is None:
                                    best_labels = labels
                
                elif cluster_method == 'dbscan':
                    # Try different eps values
                    eps_values = [0.5, 0.6, 0.7, 0.8]
                    for eps in eps_values:
                        clustering = DBSCAN(eps=eps, min_samples=1)
                        labels = clustering.fit_predict(embeddings_array)
                        
                        if len(set(labels)) > 1 and -1 not in labels:  # Valid clustering
                            try:
                                score = silhouette_score(embeddings_array, labels)
                                if score > best_score:
                                    best_score = score
                                    best_labels = labels
                            except:
                                # If silhouette score fails, use this clustering anyway
                                if best_labels is None:
                                    best_labels = labels
                
                elif cluster_method == 'spectral':
                    # Try different numbers of clusters
                    max_clusters = min(8, len(embeddings) // 2)
                    for n_clusters in range(2, max_clusters + 1):
                        clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
                        labels = clustering.fit_predict(embeddings_array)
                        
                        if len(set(labels)) > 1:
                            try:
                                score = silhouette_score(embeddings_array, labels)
                                if score > best_score:
                                    best_score = score
                                    best_labels = labels
                            except:
                                # If silhouette score fails, use this clustering anyway
                                if best_labels is None:
                                    best_labels = labels
                        
            except Exception as e:
                print(f"Warning: {cluster_method} clustering failed: {e}")
                continue
        
        # Fallback to single cluster if no good clustering found
        if best_labels is None:
            best_labels = np.array([0] * len(embeddings))
        
        # Ensure we return a list
        if hasattr(best_labels, 'tolist'):
            return best_labels.tolist()
        else:
            return list(best_labels)
    
    def assign_speaker_names(self, labels: List[int]) -> List[str]:
        """Assign human-readable names to speaker clusters"""
        unique_labels = list(set(labels))
        speaker_map = {}
        
        for i, label in enumerate(unique_labels):
            if i < len(self.speaker_names):
                speaker_map[label] = self.speaker_names[i]
            else:
                speaker_map[label] = f"Speaker {label + 1}"
        
        return [speaker_map[label] for label in labels]
    
    def diarize(self, audio_segments: List[np.ndarray], method: str = 'auto') -> Tuple[List[int], List[str]]:
        """Perform complete speaker diarization"""
        if not audio_segments:
            return [], []
        
        # Extract embeddings
        embeddings = self.extract_embeddings(audio_segments)
        
        # Cluster speakers
        labels = self.cluster_speakers(embeddings, method)
        
        # Assign names
        speaker_names = self.assign_speaker_names(labels)
        
        return labels, speaker_names


class AudioVideoConverter:
    """Convert various audio and video formats to WAV for transcription"""
    
    def __init__(self):
        self.supported_formats = {
            # Audio formats
            'audio': ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.opus', '.aiff', '.au'],
            # Video formats  
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts', '.mts', '.m2ts'],
            # All supported extensions
            'all': ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.opus', '.aiff', '.au',
                   '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts', '.mts', '.m2ts']
        }
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_formats['all']
    
    def is_video_file(self, file_path: str) -> bool:
        """Check if file is a video format"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_formats['video']
    
    def convert_to_wav(self, input_path: str, output_path: str = None, progress_callback=None) -> str:
        """
        Convert audio/video file to WAV format using ffmpeg
        
        Args:
            input_path: Path to input file
            output_path: Path for output WAV file (optional, creates temp file if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to converted WAV file
        """
        if output_path is None:
            # Create temporary WAV file
            temp_dir = tempfile.gettempdir()
            temp_name = f"whisper_temp_{int(time.time())}.wav"
            output_path = os.path.join(temp_dir, temp_name)
        
        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                output_path
            ]
            
            # Run ffmpeg
            if progress_callback:
                progress_callback(10, "Converting audio/video to WAV...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if progress_callback:
                progress_callback(20, "Conversion completed")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        except Exception as e:
            raise RuntimeError(f"Conversion error: {str(e)}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about the audio/video file"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant info
            format_info = info.get('format', {})
            streams = info.get('streams', [])
            
            # Find audio stream
            audio_stream = None
            video_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                elif stream.get('codec_type') == 'video':
                    video_stream = stream
            
            return {
                'duration': float(format_info.get('duration', 0)),
                'size_mb': float(format_info.get('size', 0)) / (1024 * 1024),
                'has_audio': audio_stream is not None,
                'has_video': video_stream is not None,
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'video_codec': video_stream.get('codec_name') if video_stream else None,
                'sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
                'channels': int(audio_stream.get('channels', 0)) if audio_stream else None
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'duration': 0,
                'size_mb': 0,
                'has_audio': False,
                'has_video': False
            }


class SentenceTranscriptionWorker(QThread):
    """Worker thread for sentence-by-sentence transcription like demo_sentence_transcription from main.py"""
    transcription_ready = pyqtSignal(str)  # new sentence text
    error_occurred = pyqtSignal(str)
    
    def __init__(self, whisper_model: WhisperModel):
        super().__init__()
        self.whisper_model = whisper_model
        self.running = False
        self.sentence_queue = queue.Queue()
        
    def run(self):
        """Process sentences from queue - async like main.py"""
        self.running = True
        while self.running:
            try:
                sentence = self.sentence_queue.get(timeout=1.0)
                result = self.whisper_model.transcribe_array(sentence, 16000)
                text = result['text'].strip()
                if text:
                    self.transcription_ready.emit(text)
            except queue.Empty:
                continue
            except Exception as e:
                self.error_occurred.emit(str(e))
                
    def add_sentence(self, sentence):
        """Add sentence for processing."""
        self.sentence_queue.put(sentence)
        
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.clear_queue()
        
    def clear_queue(self):
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except queue.Empty:
                break


class AudioRecorderWorker(QThread):
    """Worker thread for audio recording with sentence detection like demo_sentence_transcription"""
    sentence_detected = pyqtSignal(object)  # sentence audio data
    error_occurred = pyqtSignal(str)
    
    def __init__(self, device_index: Optional[int] = None, silence_threshold: float = 0.003, 
                 min_silence_duration: float = 0.8, min_sentence_duration: float = 0.7, level_callback: Optional[callable] = None):
        super().__init__()
        self.device_index = device_index
        self.running = False
        self.audio_buffer = []
        self.last_non_silent = time.time()
        self.level_callback = level_callback
        # Parameters from demo_sentence_transcription (now configurable)
        self.target_sample_rate = 16000
        self.channels = 1
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.min_sentence_duration = min_sentence_duration
        self.chunk_size = 1024
        self.actual_sample_rate = None
    
    def run(self):
        """Record audio and detect sentences, auto-try sample rates and resample if needed"""
        self.running = True
        self.audio_buffer = []
        self.last_non_silent = time.time()
        sample_rates_to_try = [16000, 44100, 48000]
        stream = None
        for sr in sample_rates_to_try:
            try:
                if self.device_index is not None:
                    sd.default.device[0] = self.device_index
                self.actual_sample_rate = sr
                stream = sd.InputStream(
                    samplerate=sr,
                    channels=self.channels,
                    callback=self._make_audio_callback(sr),
                    dtype='int16',
                    blocksize=self.chunk_size
                )
                stream.start()
                break
            except Exception as e:
                stream = None
                continue
        if stream is None:
            self.error_occurred.emit("Could not open input stream at 16k, 44.1k, or 48kHz. Please check your device settings.")
            return
        try:
            while self.running:
                time.sleep(0.1)
        finally:
            stream.stop()
            stream.close()
    
    def _make_audio_callback(self, sr):
        def audio_callback(indata, frames, t, status):
            if status:
                print(f"\r⚠️  Audio status: {status}", end="", flush=True)
            if not self.running:
                return
            # Sensitivity bar update
            if self.level_callback:
                level = int(min(100, float(np.abs(indata).max()) / 32767.0 * 100))
                self.level_callback(level)
            # Convert to float32
            samples = indata.flatten().astype(np.float32) / 32768.0
            # Resample if needed
            if sr != self.target_sample_rate:
                samples = scipy.signal.resample(samples, int(len(samples) * self.target_sample_rate / sr))
            self.audio_buffer.extend(samples)
            # Compute RMS energy
            rms = np.sqrt(np.mean(samples ** 2))
            now = time.time()
            if rms > self.silence_threshold:
                self.last_non_silent = now
            # If enough silence and enough audio, treat as sentence
            if (now - self.last_non_silent > self.min_silence_duration and
                len(self.audio_buffer) > int(self.min_sentence_duration * self.target_sample_rate)):
                buffer_rms = np.sqrt(np.mean(np.array(self.audio_buffer) ** 2))
                if buffer_rms > self.silence_threshold:
                    sentence = np.array(self.audio_buffer, dtype=np.float32)
                    self.sentence_detected.emit(sentence)
                self.audio_buffer = []
        return audio_callback
    
    def stop(self):
        """Stop recording."""
        self.running = False


class FileTranscriptionWorker(QThread):
    """Worker thread for file transcription with format conversion support"""
    progress_updated = pyqtSignal(int)  # progress percentage
    status_updated = pyqtSignal(str)  # status message
    transcription_ready = pyqtSignal(str, dict)  # text, metadata
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, whisper_model: WhisperModel, file_path: str):
        super().__init__()
        self.whisper_model = whisper_model
        self.file_path = file_path
        self.segment_duration = 20  # seconds
        self.overlap_ratio = 0.5
        self.sample_rate = 16000
        self.converter = AudioVideoConverter()
        self.temp_wav_path = None
        
    def run(self):
        try:
            # Check if file format is supported
            if not self.converter.is_supported_format(self.file_path):
                self.error_occurred.emit(f"Unsupported file format: {Path(self.file_path).suffix}")
                self.finished.emit()
                return
            
            # Get file info
            self.status_updated.emit("Analyzing file...")
            file_info = self.converter.get_file_info(self.file_path)
            
            if 'error' in file_info:
                self.error_occurred.emit(f"Error analyzing file: {file_info['error']}")
                self.finished.emit()
                return
            
            if not file_info['has_audio']:
                self.error_occurred.emit("No audio stream found in file")
                self.finished.emit()
                return
            
            # Show file info
            duration_str = f"{file_info['duration']:.1f}s" if file_info['duration'] > 0 else "Unknown"
            size_str = f"{file_info['size_mb']:.1f}MB" if file_info['size_mb'] > 0 else "Unknown"
            self.status_updated.emit(f"File: {duration_str}, {size_str}")
            
            # Convert to WAV if needed
            if Path(self.file_path).suffix.lower() != '.wav':
                self.status_updated.emit("Converting to WAV format...")
                self.temp_wav_path = self.converter.convert_to_wav(
                    self.file_path, 
                    progress_callback=lambda p, msg: self.status_updated.emit(msg)
                )
                audio_file_path = self.temp_wav_path
            else:
                audio_file_path = self.file_path
            
            # Read audio file
            self.status_updated.emit("Loading audio...")
            audio, sr = sf.read(audio_file_path)
            
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono if needed
            
            # Ensure correct sample rate
            if sr != self.sample_rate:
                self.status_updated.emit("Resampling audio...")
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
                except ImportError:
                    self.error_occurred.emit("Audio file sample rate does not match model and librosa is not installed for resampling.")
                    self.finished.emit()
                    return
            
            # Segment audio for processing
            self.status_updated.emit("Segmenting audio...")
            segment_samples = int(self.segment_duration * self.sample_rate)
            overlap_samples = int(segment_samples * self.overlap_ratio)
            step = segment_samples - overlap_samples
            total_samples = len(audio)
            segments = []
            
            for start in range(0, total_samples - segment_samples + 1, step):
                segment = audio[start:start+segment_samples]
                segments.append(segment)
            
            if total_samples % step != 0 and total_samples > segment_samples:
                # Add last segment if not already included
                segment = audio[-segment_samples:]
                segments.append(segment)
            
            n_segments = len(segments)
            transcripts = []
            
            # Transcribe segments
            self.status_updated.emit("Transcribing segments...")
            for i, segment in enumerate(segments):
                segment = np.asarray(segment).flatten().astype(np.float32)
                MAX_SEGMENT_SECONDS = 20
                if len(segment) > self.sample_rate * MAX_SEGMENT_SECONDS:
                    segment = segment[-(self.sample_rate * MAX_SEGMENT_SECONDS):]
                
                result = self.whisper_model.transcribe_segment(segment)
                # Add audio data to result for diarization
                result['audio'] = segment
                transcripts.append(result)
                
                percent = int(20 + 70 * (i+1) / n_segments)  # 20-90% for transcription
                self.progress_updated.emit(percent)
                self.status_updated.emit(f"Transcribing segment {i+1}/{n_segments}")
            
            # Merge transcripts
            self.status_updated.emit("Merging transcripts...")
            merged_text = self.whisper_model.merge_transcripts(transcripts)
            
            # Add file info to metadata
            metadata = {
                "segments": transcripts,
                "file_info": file_info,
                "original_file": self.file_path,
                "converted_file": self.temp_wav_path
            }
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Transcription completed")
            self.transcription_ready.emit(merged_text, metadata)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Clean up temporary file
            if self.temp_wav_path and os.path.exists(self.temp_wav_path):
                try:
                    os.remove(self.temp_wav_path)
                except:
                    pass  # Ignore cleanup errors
            self.finished.emit()


class WASAPILoopbackRecorderWorker(QThread):
    sentence_detected = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    def __init__(self, silence_threshold=0.003, min_silence_duration=0.8, min_sentence_duration=0.7, level_callback=None):
        super().__init__()
        self.running = False
        self.audio_buffer = []
        self.last_non_silent = time.time()
        self.level_callback = level_callback
        self.target_sample_rate = 16000
        self.loopback_sample_rate = 44100  # or 48000
        self.channels = 1
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.min_sentence_duration = min_sentence_duration
        self.chunk_size = 1024
    def run(self):
        import scipy.signal
        self.running = True
        self.audio_buffer = []
        self.last_non_silent = time.time()
        try:
            mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
            with mic.recorder(samplerate=self.loopback_sample_rate) as rec:
                while self.running:
                    data = rec.record(numframes=self.chunk_size)
                    # If stereo, average to mono
                    if data.ndim > 1 and data.shape[1] > 1:
                        data = data.mean(axis=1)
                    samples = data.astype(np.float32)
                    # Normalize to -1.0..1.0
                    if np.abs(samples).max() > 1.0:
                        samples = samples / np.abs(samples).max()
                    # High-quality resample
                    if self.loopback_sample_rate != self.target_sample_rate:
                        samples = scipy.signal.resample_poly(samples, self.target_sample_rate, self.loopback_sample_rate)
                    if self.level_callback:
                        level = int(min(100, float(np.abs(samples).max()) * 100))
                        self.level_callback(level)
                    self.audio_buffer.extend(samples)
                    rms = np.sqrt(np.mean(samples ** 2))
                    now = time.time()
                    if rms > self.silence_threshold:
                        self.last_non_silent = now
                    if (now - self.last_non_silent > self.min_silence_duration and
                        len(self.audio_buffer) > int(self.min_sentence_duration * self.target_sample_rate)):
                        buffer_rms = np.sqrt(np.mean(np.array(self.audio_buffer) ** 2))
                        if buffer_rms > self.silence_threshold:
                            sentence = np.array(self.audio_buffer, dtype=np.float32)
                            self.sentence_detected.emit(sentence)
                        self.audio_buffer = []
        except Exception as e:
            self.error_occurred.emit(str(e))
    def stop(self):
        self.running = False


class LiveTranscriptionTab(QWidget):
    """Tab for live transcription mode with sentence detection like demo_sentence_transcription."""
    
    def __init__(self, whisper_model: WhisperModel):
        super().__init__()
        self.whisper_model = whisper_model
        self.audio_recorder = None
        self.transcription_worker = None
        self.selected_device_index = None
        self.full_transcript = ""
        self.speaker_diarization_enabled = False
        self.sentence_audio_list = []  # (audio, text, embedding)
        self.diarizer = SpeakerDiarizer()
        self._transcript_buffer = None
        self._transcript_update_timer = QTimer(self)
        self._transcript_update_timer.setInterval(100)  # ms
        self._transcript_update_timer.setSingleShot(True)
        self._transcript_update_timer.timeout.connect(self._apply_transcript_update)
        self.setup_ui()
        self.setup_connections()
        self.update_device_list()
        
    def _debounced_update_transcript(self, text: str):
        self._transcript_buffer = text
        if not self._transcript_update_timer.isActive():
            self._transcript_update_timer.start()

    def _apply_transcript_update(self):
        if self._transcript_buffer is not None:
            self.transcript_display.setPlainText(self._transcript_buffer)
            scrollbar = self.transcript_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            self._transcript_buffer = None

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Device selection and sensitivity
        device_group = QGroupBox("Audio Input")
        device_layout = QVBoxLayout()
        
        # Device selection row
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        device_row.addWidget(QLabel("Input Device:"))
        device_row.addWidget(self.device_combo)
        
        # Refresh button for devices
        self.refresh_devices_button = QPushButton("🔄")
        self.refresh_devices_button.setToolTip("Refresh audio device list")
        self.refresh_devices_button.setFixedWidth(30)
        self.refresh_devices_button.clicked.connect(self.update_device_list)
        device_row.addWidget(self.refresh_devices_button)
        
        # Add Detect Active Devices button below device selection
        self.detect_active_button = QPushButton("Detect Active Devices")
        self.detect_active_button.setToolTip("Scan all input devices for audio activity")
        self.detect_active_button.clicked.connect(self.detect_active_devices)
        device_row.addWidget(self.detect_active_button)
        
        device_layout.addLayout(device_row)
        
        # Sensitivity and info row
        info_row = QHBoxLayout()
        self.sensitivity_bar = QProgressBar()
        self.sensitivity_bar.setRange(0, 100)
        self.sensitivity_bar.setTextVisible(False)
        self.sensitivity_bar.setFixedWidth(150)
        info_row.addWidget(QLabel("Input Sensitivity:"))
        info_row.addWidget(self.sensitivity_bar)
        info_row.addStretch()
        
        # System audio info
        system_audio_info = QLabel("💡 For meetings/videos: Enable 'Stereo Mix' in Windows Sound settings")
        system_audio_info.setStyleSheet("QLabel { color: white; font-size: 10px; }")
        system_audio_info.setWordWrap(True)
        info_row.addWidget(system_audio_info)
        
        # Help button for system audio
        self.system_audio_help_button = QPushButton("?")
        self.system_audio_help_button.setToolTip("Help with system audio capture")
        self.system_audio_help_button.setFixedWidth(25)
        self.system_audio_help_button.clicked.connect(self.show_system_audio_help)
        info_row.addWidget(self.system_audio_help_button)
        
        device_layout.addLayout(info_row)
        
        device_group.setLayout(device_layout)
        
        # Sentence detection parameters
        params_group = QGroupBox("Sentence Detection Parameters")
        params_layout = QGridLayout()
        
        # Info label
        info_label = QLabel("Adjust these parameters to fine-tune sentence detection sensitivity:")
        info_label.setStyleSheet("QLabel { color: white; font-style: italic; }")
        params_layout.addWidget(info_label, 0, 0, 1, 4)
        
        # Silence threshold
        self.silence_threshold_spin = QDoubleSpinBox()
        self.silence_threshold_spin.setRange(0.001, 0.1)
        self.silence_threshold_spin.setSingleStep(0.001)
        self.silence_threshold_spin.setDecimals(3)
        self.silence_threshold_spin.setValue(0.003)
        self.silence_threshold_spin.setToolTip("RMS threshold for detecting silence (lower = more sensitive)")
        params_layout.addWidget(QLabel("Silence Threshold:"), 1, 0)
        params_layout.addWidget(self.silence_threshold_spin, 1, 1)
        
        # Min silence duration
        self.min_silence_duration_spin = QDoubleSpinBox()
        self.min_silence_duration_spin.setRange(0.1, 3.0)
        self.min_silence_duration_spin.setSingleStep(0.1)
        self.min_silence_duration_spin.setDecimals(1)
        self.min_silence_duration_spin.setValue(0.8)
        self.min_silence_duration_spin.setToolTip("Minimum silence duration to end a sentence (seconds)")
        params_layout.addWidget(QLabel("Min Silence Duration (s):"), 1, 2)
        params_layout.addWidget(self.min_silence_duration_spin, 1, 3)
        
        # Min sentence duration
        self.min_sentence_duration_spin = QDoubleSpinBox()
        self.min_sentence_duration_spin.setRange(0.1, 2.0)
        self.min_sentence_duration_spin.setSingleStep(0.1)
        self.min_sentence_duration_spin.setDecimals(1)
        self.min_sentence_duration_spin.setValue(0.7)
        self.min_sentence_duration_spin.setToolTip("Minimum duration for a valid sentence (seconds)")
        params_layout.addWidget(QLabel("Min Sentence Duration (s):"), 2, 0)
        params_layout.addWidget(self.min_sentence_duration_spin, 2, 1)
        
        # Reset to defaults button
        self.reset_params_button = QPushButton("Reset to Defaults")
        self.reset_params_button.setStyleSheet("QPushButton { padding: 4px; }")
        self.reset_params_button.clicked.connect(self.reset_parameters_to_defaults)
        params_layout.addWidget(self.reset_params_button, 2, 2, 1, 2)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Speaker diarization settings
        diarization_group = QGroupBox("Speaker Diarization")
        diarization_layout = QGridLayout()
        
        # Differentiate Speakers checkbox
        self.speaker_checkbox = QCheckBox("Differentiate Speakers")
        self.speaker_checkbox.setToolTip("Label transcript lines by speaker (diarization)")
        self.speaker_checkbox.stateChanged.connect(self.on_speaker_checkbox_changed)
        diarization_layout.addWidget(self.speaker_checkbox, 0, 0, 1, 2)
        
        # Clustering method selection
        diarization_layout.addWidget(QLabel("Clustering Method:"), 1, 0)
        self.clustering_method_combo = QComboBox()
        self.clustering_method_combo.addItems(["Auto (Best)", "Agglomerative", "DBSCAN", "Spectral"])
        self.clustering_method_combo.setToolTip("Algorithm for grouping similar voices")
        diarization_layout.addWidget(self.clustering_method_combo, 1, 1)
        
        # Speaker info
        speaker_info = QLabel("🎤 Uses AI to identify different speakers and assign names (Alice, Bob, etc.)")
        speaker_info.setStyleSheet("QLabel { color: white; font-size: 10px; }")
        speaker_info.setWordWrap(True)
        diarization_layout.addWidget(speaker_info, 2, 0, 1, 2)
        
        diarization_group.setLayout(diarization_layout)
        layout.addWidget(diarization_group)
        
        # Control panel
        control_group = QGroupBox("Recording Controls")
        control_layout = QGridLayout()
        
        # Start/Stop button
        self.start_button = QPushButton("Start Recording")
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        control_layout.addWidget(self.start_button, 0, 0)
        
        # Pause/Resume button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 8px; }")
        control_layout.addWidget(self.pause_button, 0, 1)
        
        # Stop button
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 8px; }")
        control_layout.addWidget(self.stop_button, 0, 2)
        
        # Clear button
        self.clear_button = QPushButton("Clear Transcript")
        self.clear_button.setStyleSheet("QPushButton { padding: 8px; }")
        control_layout.addWidget(self.clear_button, 0, 3)
        
        control_group.setLayout(control_layout)
        layout.addWidget(device_group)
        layout.addWidget(control_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to record")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; color: white; }")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Transcript display
        transcript_group = QGroupBox("Live Transcription")
        transcript_layout = QVBoxLayout()
        
        self.transcript_display = QTextEdit()
        self.transcript_display.setFont(QFont("Consolas", 10))
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setMinimumHeight(300)
        transcript_layout.addWidget(self.transcript_display)
        
        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group)
        
        self.setLayout(layout)
        
    def setup_connections(self):
        """Setup signal connections"""
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        self.start_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_recording)
        self.clear_button.clicked.connect(self.clear_transcript)
        
    def update_device_list(self):
        """Update the list of available audio devices including system audio capture"""
        self.device_combo.clear()
        devices = sd.query_devices()
        
        # Keywords that indicate system audio capture devices
        system_audio_keywords = [
            'stereo mix', 'what u hear', 'wave out mix', 'monitor mix',
            'system audio', 'loopback', 'virtual audio', 'cable output',
            'vb-audio', 'voicemeeter', 'audacity', 'system capture'
        ]
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Only input devices
                name = device['name'].lower()
                device_name = device['name']
                
                # Check if this is a system audio capture device
                is_system_audio = any(keyword in name for keyword in system_audio_keywords)
                
                # Add device with appropriate label
                if i == sd.default.device[0]:
                    device_name += " (Default)"
                elif is_system_audio:
                    device_name += " (System Audio)"
                
                self.device_combo.addItem(device_name, i)
                
                # Add tooltip for system audio devices
                if is_system_audio:
                    last_index = self.device_combo.count() - 1
                    self.device_combo.setItemData(last_index, 
                        "Use this device to capture audio from your computer's speakers (meetings, videos, etc.)", 
                        Qt.ItemDataRole.ToolTipRole)
                
        # Select default device
        default_idx = sd.default.device[0]
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == default_idx:
                self.device_combo.setCurrentIndex(i)
                break
                
        # Show message if no devices found
        if self.device_combo.count() == 0:
            self.device_combo.addItem("No audio input devices found", -1)

        # Add the loopback option at the end
        self.device_combo.addItem("Default Output Device (Loopback)", "__loopback__")

    def on_device_changed(self, idx):
        """Handle device selection change"""
        if idx >= 0:
            self.selected_device_index = self.device_combo.itemData(idx)

    def start_recording(self):
        """Start live recording and transcription with sentence detection."""
        try:
            self.transcription_worker = SentenceTranscriptionWorker(self.whisper_model)
            self.transcription_worker.transcription_ready.connect(self.on_transcription_ready)
            self.transcription_worker.error_occurred.connect(self.on_transcription_error)
            self.transcription_worker.start()
            if self.device_combo.currentData() == "__loopback__":
                self.audio_recorder = WASAPILoopbackRecorderWorker(
                    self.silence_threshold_spin.value(),
                    self.min_silence_duration_spin.value(),
                    self.min_sentence_duration_spin.value(),
                    level_callback=self.update_sensitivity_bar
                )
            else:
                self.audio_recorder = AudioRecorderWorker(
                    self.selected_device_index,
                    self.silence_threshold_spin.value(),
                    self.min_silence_duration_spin.value(),
                    self.min_sentence_duration_spin.value(),
                    level_callback=self.update_sensitivity_bar
                )
            self.audio_recorder.sentence_detected.connect(self.on_sentence_detected)
            self.audio_recorder.error_occurred.connect(self.on_transcription_error)
            self.audio_recorder.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.status_label.setText("Recording... Speak a sentence, pause, and see the result.")
            self.status_label.setStyleSheet("QLabel { font-weight: bold; color: #4CAF50; }")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording: {str(e)}")
            
    def toggle_pause(self):
        """Toggle pause/resume recording"""
        if hasattr(self, 'audio_recorder') and self.audio_recorder:
            if self.audio_recorder.running:
                self.audio_recorder.stop()
                self.pause_button.setText("Resume")
                self.status_label.setText("Paused")
            else:
                self.audio_recorder.start()
                self.pause_button.setText("Pause")
                self.status_label.setText("Recording...")
            
    def stop_recording(self):
        """Stop recording and transcription."""
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder.wait()
            self.audio_recorder = None
            
        if self.transcription_worker:
            self.transcription_worker.stop()
            self.transcription_worker.wait()
            self.transcription_worker = None
            
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.status_label.setText("Recording stopped")
        self.status_label.setStyleSheet("QLabel { font-weight: bold; color: #666; }")
        self.sensitivity_bar.setValue(0)
        
    def clear_transcript(self):
        """Clear the transcript display and reset."""
        self.transcript_display.clear()
        self.full_transcript = ""
        self.sentence_audio_list.clear()  # Clear the audio segments list
        if self.transcription_worker:
            self.transcription_worker.clear_queue()
        # Stop the debounce timer if it's running
        if self._transcript_update_timer.isActive():
            self._transcript_update_timer.stop()
        self._transcript_buffer = None

    def on_sentence_detected(self, sentence_audio):
        """Handle detected sentence from audio recorder."""
        if self.transcription_worker:
            self.transcription_worker.add_sentence(sentence_audio)
        if self.speaker_diarization_enabled:
            self.sentence_audio_list.append((sentence_audio, None, None))
            
    def on_transcription_ready(self, new_text: str):
        """Handle transcription result."""
        if self.speaker_diarization_enabled:
            # Find the last sentence_audio with None text, fill it
            for i in range(len(self.sentence_audio_list)-1, -1, -1):
                if self.sentence_audio_list[i][1] is None:
                    audio = self.sentence_audio_list[i][0]
                    self.sentence_audio_list[i] = (audio, new_text, None)  # Will get embedding later
                    break
            # Get clustering method
            method_map = {
                "Auto (Best)": "auto",
                "Agglomerative": "agglomerative", 
                "DBSCAN": "dbscan",
                "Spectral": "spectral"
            }
            clustering_method = method_map.get(self.clustering_method_combo.currentText(), "auto")
            # Perform diarization
            audio_segments = [item[0] for item in self.sentence_audio_list]
            labels, speaker_names = self.diarizer.diarize(audio_segments, method=clustering_method)
            # Format transcript with speaker names
            lines = []
            for (audio, text, emb), speaker_name in zip(self.sentence_audio_list, speaker_names):
                if text:  # Only show lines with actual text
                    lines.append(f"{speaker_name}: {text}")
            self.full_transcript = "\n".join(lines)
            self._debounced_update_transcript(self.full_transcript)
            return
        # Default behavior
        self.full_transcript += " " + new_text
        self.full_transcript = self.full_transcript.strip()
        self._debounced_update_transcript(self.full_transcript)
        
    def on_transcription_error(self, error: str):
        """Handle transcription error"""
        QMessageBox.warning(self, "Transcription Error", f"Error: {error}")

    def update_sensitivity_bar(self, level_val: int):
        """Update sensitivity bar from audio handler callback (value 0-100)"""
        self.sensitivity_bar.setValue(level_val)

    def reset_parameters_to_defaults(self):
        """Reset sentence detection parameters to defaults"""
        self.silence_threshold_spin.setValue(0.003)
        self.min_silence_duration_spin.setValue(0.8)
        self.min_sentence_duration_spin.setValue(0.7)

    def is_system_audio_device(self, device_name: str) -> bool:
        """Check if a device is a system audio capture device"""
        name_lower = device_name.lower()
        system_audio_keywords = [
            'stereo mix', 'what u hear', 'wave out mix', 'monitor mix',
            'system audio', 'loopback', 'virtual audio', 'cable output',
            'vb-audio', 'voicemeeter', 'audacity', 'system capture'
        ]
        return any(keyword in name_lower for keyword in system_audio_keywords)

    def show_system_audio_help(self):
        """Show help dialog for enabling system audio capture"""
        help_text = """
To capture system audio (meetings, videos, etc.):

1. Enable 'Stereo Mix' in Windows:
   • Right-click speaker icon → Sound settings
   • Click 'Sound Control Panel'
   • Go to Recording tab
   • Right-click empty space → 'Show Disabled Devices'
   • Enable 'Stereo Mix' or similar device

2. Alternative: Install virtual audio software:
   • VB-Audio Virtual Cable
   • Voicemeeter
   • Audacity (for recording)

3. Select the system audio device from the dropdown above

Note: System audio capture allows you to transcribe:
• Video calls (Zoom, Teams, etc.)
• YouTube videos
• Music and podcasts
• Any audio playing through your speakers
        """
        
        QMessageBox.information(self, "System Audio Capture Help", help_text.strip())

    def detect_active_devices(self):
        import sounddevice as sd
        import numpy as np
        from PyQt6.QtWidgets import QMessageBox
        devices = sd.query_devices()
        active_devices = []
        scan_duration = 0.5  # seconds
        sample_rates = [16000, 44100, 48000]
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                for sr in sample_rates:
                    try:
                        stream = sd.InputStream(device=i, samplerate=sr, channels=1, dtype='int16', blocksize=1024)
                        stream.start()
                        audio = stream.read(int(sr * scan_duration))[0]
                        stream.stop()
                        stream.close()
                        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2)) / 32768.0
                        if rms > 0.01:  # threshold for "active"
                            active_devices.append(f"[{i}] {device['name']} (RMS={rms:.3f}, {sr}Hz)")
                        break
                    except Exception:
                        continue
        if active_devices:
            msg = "Devices currently receiving audio (media/mic):\n\n" + "\n".join(active_devices)
        else:
            msg = "No input devices are currently receiving significant audio.\n\nTry playing media or speaking into your mic, then scan again."
        QMessageBox.information(self, "Active Input Devices", msg)

    def on_speaker_checkbox_changed(self, state):
        self.speaker_diarization_enabled = bool(state)
        if not self.speaker_diarization_enabled:
            self.sentence_audio_list.clear()
            self.full_transcript = ""
            self.transcript_display.clear()
            # Stop the debounce timer if it's running
            if self._transcript_update_timer.isActive():
                self._transcript_update_timer.stop()
            self._transcript_buffer = None


class FileTranscriptionTab(QWidget):
    """Tab for file transcription mode"""
    
    def __init__(self, whisper_model: WhisperModel):
        super().__init__()
        self.whisper_model = whisper_model
        self.file_worker = None
        self.speaker_diarization_enabled = False
        self.diarizer = SpeakerDiarizer()
        self._transcript_buffer = None
        self._transcript_update_timer = QTimer(self)
        self._transcript_update_timer.setInterval(100)  # ms
        self._transcript_update_timer.setSingleShot(True)
        self._transcript_update_timer.timeout.connect(self._apply_transcript_update)
        self.setup_ui()
        self.setup_connections()
        
    def _debounced_update_transcript(self, text: str):
        self._transcript_buffer = text
        if not self._transcript_update_timer.isActive():
            self._transcript_update_timer.start()

    def _apply_transcript_update(self):
        if self._transcript_buffer is not None:
            self.transcript_display.setPlainText(self._transcript_buffer)
            self._transcript_buffer = None

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # File path display
        file_path_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { padding: 8px; border: 1px solid #ccc; background: #f9f9f9; }")
        file_path_layout.addWidget(self.file_path_label, 1)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setStyleSheet("QPushButton { padding: 8px; }")
        file_path_layout.addWidget(self.browse_button)
        
        file_layout.addLayout(file_path_layout)
        
        # File info display
        self.file_info_label = QLabel("")
        self.file_info_label.setStyleSheet("QLabel { color: white; font-size: 10px; }")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("QLabel { font-weight: bold; color: white; }")
        progress_layout.addWidget(self.progress_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Speaker diarization settings
        diarization_group = QGroupBox("Speaker Diarization")
        diarization_layout = QGridLayout()
        
        # Differentiate Speakers checkbox
        self.speaker_checkbox = QCheckBox("Differentiate Speakers")
        self.speaker_checkbox.setToolTip("Label transcript lines by speaker (diarization)")
        self.speaker_checkbox.stateChanged.connect(self.on_speaker_checkbox_changed)
        diarization_layout.addWidget(self.speaker_checkbox, 0, 0, 1, 2)
        
        # Clustering method selection
        diarization_layout.addWidget(QLabel("Clustering Method:"), 1, 0)
        self.clustering_method_combo = QComboBox()
        self.clustering_method_combo.addItems(["Auto (Best)", "Agglomerative", "DBSCAN", "Spectral"])
        self.clustering_method_combo.setToolTip("Algorithm for grouping similar voices")
        diarization_layout.addWidget(self.clustering_method_combo, 1, 1)
        
        # Speaker info
        speaker_info = QLabel("🎤 Uses AI to identify different speakers and assign names (Alice, Bob, etc.)")
        speaker_info.setStyleSheet("QLabel { color: white; font-size: 10px; }")
        speaker_info.setWordWrap(True)
        diarization_layout.addWidget(speaker_info, 2, 0, 1, 2)
        
        diarization_group.setLayout(diarization_layout)
        layout.addWidget(diarization_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.transcribe_button = QPushButton("Transcribe File")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 8px; }")
        control_layout.addWidget(self.transcribe_button)
        
        self.save_button = QPushButton("Save Transcript")
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("QPushButton { padding: 8px; }")
        control_layout.addWidget(self.save_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet("QPushButton { padding: 8px; }")
        control_layout.addWidget(self.clear_button)
        
        layout.addLayout(control_layout)
        
        # Supported formats info
        formats_group = QGroupBox("Supported Formats")
        formats_layout = QVBoxLayout()
        
        formats_text = """
        <b>Audio:</b> MP3, WAV, M4A, AAC, OGG, FLAC, WMA, OPUS, AIFF, AU
        <b>Video:</b> MP4, AVI, MOV, MKV, WMV, FLV, WEBM, M4V, 3GP, TS, MTS, M2TS
        
        <i>All formats are automatically converted to WAV for transcription.</i>
        """
        formats_label = QLabel(formats_text)
        formats_label.setStyleSheet("QLabel { color: white; font-size: 10px; }")
        formats_label.setWordWrap(True)
        formats_layout.addWidget(formats_label)
        
        formats_group.setLayout(formats_layout)
        layout.addWidget(formats_group)
        
        # Transcript display
        transcript_group = QGroupBox("Transcription Result")
        transcript_layout = QVBoxLayout()
        
        self.transcript_display = QTextEdit()
        self.transcript_display.setFont(QFont("Consolas", 10))
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setMinimumHeight(300)
        transcript_layout.addWidget(self.transcript_display)
        
        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group)
        
        self.setLayout(layout)
        
    def setup_connections(self):
        """Setup signal connections"""
        self.browse_button.clicked.connect(self.browse_file)
        self.transcribe_button.clicked.connect(self.transcribe_file)
        self.save_button.clicked.connect(self.save_transcript)
        self.clear_button.clicked.connect(self.clear_transcript)
        
    def browse_file(self):
        """Browse for audio/video file with extended format support"""
        # Create comprehensive file filter
        audio_formats = "*.mp3 *.wav *.m4a *.aac *.ogg *.flac *.wma *.opus *.aiff *.au"
        video_formats = "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp *.ts *.mts *.m2ts"
        all_formats = f"{audio_formats} {video_formats}"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio/Video File",
            "",
            f"All Supported Files ({all_formats});;"
            f"Audio Files ({audio_formats});;"
            f"Video Files ({video_formats});;"
            f"MP3 Files (*.mp3);;"
            f"WAV Files (*.wav);;"
            f"MP4 Files (*.mp4);;"
            f"All Files (*)"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.transcribe_button.setEnabled(True)
            self.display_file_info(file_path)
            
    def display_file_info(self, file_path: str):
        """Display information about the selected file"""
        try:
            converter = AudioVideoConverter()
            if not converter.is_supported_format(file_path):
                self.file_info_label.setText("⚠️ Unsupported file format")
                self.transcribe_button.setEnabled(False)
                return
            
            file_info = converter.get_file_info(file_path)
            
            if 'error' in file_info:
                self.file_info_label.setText(f"⚠️ Error analyzing file: {file_info['error']}")
                return
            
            # Build info string
            info_parts = []
            
            if file_info['has_video']:
                info_parts.append("🎬 Video file")
            else:
                info_parts.append("🎵 Audio file")
            
            if file_info['duration'] > 0:
                minutes = int(file_info['duration'] // 60)
                seconds = int(file_info['duration'] % 60)
                info_parts.append(f"⏱️ {minutes}:{seconds:02d}")
            
            if file_info['size_mb'] > 0:
                info_parts.append(f"📁 {file_info['size_mb']:.1f}MB")
            
            if file_info['audio_codec']:
                info_parts.append(f"🎤 {file_info['audio_codec'].upper()}")
            
            if file_info['video_codec']:
                info_parts.append(f"📹 {file_info['video_codec'].upper()}")
            
            if file_info['sample_rate']:
                info_parts.append(f"🔊 {file_info['sample_rate']}Hz")
            
            if file_info['channels']:
                info_parts.append(f"🎧 {file_info['channels']}ch")
            
            if not file_info['has_audio']:
                info_parts.append("⚠️ No audio stream")
                self.transcribe_button.setEnabled(False)
            else:
                self.transcribe_button.setEnabled(True)
            
            self.file_info_label.setText(" | ".join(info_parts))
            
        except Exception as e:
            self.file_info_label.setText(f"⚠️ Error: {str(e)}")
            self.transcribe_button.setEnabled(False)

    def transcribe_file(self):
        """Start file transcription"""
        file_path = self.file_path_label.text()
        if file_path == "No file selected":
            return
            
        # Disable controls
        self.transcribe_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting transcription...")
        
        # Start worker thread
        self.file_worker = FileTranscriptionWorker(self.whisper_model, file_path)
        self.file_worker.progress_updated.connect(self.progress_bar.setValue)
        self.file_worker.status_updated.connect(self.progress_label.setText)
        self.file_worker.transcription_ready.connect(self.on_transcription_ready)
        self.file_worker.error_occurred.connect(self.on_transcription_error)
        self.file_worker.finished.connect(self.on_transcription_finished)
        self.file_worker.start()
        
    def on_transcription_ready(self, text: str, metadata: dict):
        """Handle transcription result"""
        if self.speaker_diarization_enabled and "segments" in metadata:
            segments = metadata["segments"]
            # Extract audio segments for diarization
            audio_segments = []
            for seg in segments:
                audio = seg.get("audio", None)
                if audio is not None:
                    audio_segments.append(audio)
            if len(audio_segments) > 1:
                # Get clustering method
                method_map = {
                    "Auto (Best)": "auto",
                    "Agglomerative": "agglomerative", 
                    "DBSCAN": "dbscan",
                    "Spectral": "spectral"
                }
                clustering_method = method_map.get(self.clustering_method_combo.currentText(), "auto")
                # Perform diarization
                labels, speaker_names = self.diarizer.diarize(audio_segments, method=clustering_method)
                # Format transcript with speaker names
                lines = []
                for seg, speaker_name in zip(segments, speaker_names):
                    text = seg.get('text', '').strip()
                    if text:  # Only show lines with actual text
                        lines.append(f"{speaker_name}: {text}")
                self._debounced_update_transcript("\n".join(lines))
            else:
                # Fallback to regular transcription if not enough segments
                self._debounced_update_transcript(text)
            self.save_button.setEnabled(True)
            return
        # Default behavior
        self._debounced_update_transcript(text)
        self.save_button.setEnabled(True)
        
    def on_transcription_error(self, error: str):
        """Handle transcription error"""
        QMessageBox.critical(self, "Transcription Error", f"Error: {error}")
        self.on_transcription_finished()
        
    def on_transcription_finished(self):
        """Handle transcription completion"""
        # Re-enable controls
        self.transcribe_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Ready")
        
    def save_transcript(self):
        """Save transcript to file"""
        text = self.transcript_display.toPlainText()
        if not text:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            "",
            "Text Files (*.txt);;SRT Files (*.srt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "Success", f"Transcript saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save transcript: {str(e)}")
                
    def clear_transcript(self):
        """Clear the transcript display"""
        self.transcript_display.clear()
        self.save_button.setEnabled(False)

    def on_speaker_checkbox_changed(self, state):
        self.speaker_diarization_enabled = bool(state)
        if not self.speaker_diarization_enabled:
            self.sentence_audio_list.clear()
            self.full_transcript = ""
            self.transcript_display.clear()
            # Stop the debounce timer if it's running
            if self._transcript_update_timer.isActive():
                self._transcript_update_timer.stop()
            self._transcript_buffer = None


class ModelLoaderWorker(QThread):
    model_loaded = pyqtSignal(bool)
    model_error = pyqtSignal(str)

    def __init__(self, whisper_model: WhisperModel, model_size: str):
        super().__init__()
        self.whisper_model = whisper_model
        self.model_size = model_size

    def run(self):
        try:
            success = self.whisper_model.change_model(self.model_size)
            self.model_loaded.emit(success)
        except Exception as e:
            self.model_error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.whisper_model = None
        self.setup_ui()
        self.setup_model()
        
    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Local Whisper Transcriber")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Model Size:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo, 0, 1)
        
        self.model_status_label = QLabel("Not loaded")
        self.model_status_label.setStyleSheet("QLabel { font-weight: bold; color: white; }")
        model_layout.addWidget(self.model_status_label, 0, 2)
        
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_button, 0, 3)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        central_widget.setLayout(layout)
        
    def setup_model(self):
        """Setup the Whisper model - simple like main.py"""
        self.whisper_model = WhisperModel(model_size="base")
        
    def load_model(self):
        """Load the selected model - simple like main.py"""
        model_size = self.model_combo.currentText()

        # Disable controls during loading
        self.load_model_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.model_status_label.setText("Loading...")
        self.status_bar.showMessage(f"Loading {model_size} model...")

        # Simple model creation like main.py
        self.whisper_model = WhisperModel(model_size=model_size)
        
        # Use a QThread worker for model loading
        self.model_loader_worker = ModelLoaderWorker(self.whisper_model, model_size)
        self.model_loader_worker.model_loaded.connect(self.on_model_loaded)
        self.model_loader_worker.model_error.connect(self.on_model_error)
        self.model_loader_worker.start()
        
    def on_model_loaded(self, success: bool):
        """Handle model loading completion"""
        if success:
            self.model_status_label.setText("Loaded")
            self.model_status_label.setStyleSheet("QLabel { font-weight: bold; color: #4CAF50; }")
            self.status_bar.showMessage("Model loaded successfully")
            
            # Setup tabs
            self.setup_tabs()
        else:
            self.model_status_label.setText("Failed")
            self.model_status_label.setStyleSheet("QLabel { font-weight: bold; color: #f44336; }")
            self.status_bar.showMessage("Failed to load model")
            
        # Re-enable controls
        self.load_model_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        
    def on_model_error(self, error: str):
        """Handle model loading error"""
        QMessageBox.critical(self, "Model Error", f"Failed to load model: {error}")
        self.model_status_label.setText("Error")
        self.model_status_label.setStyleSheet("QLabel { font-weight: bold; color: #f44336; }")
        
        # Re-enable controls
        self.load_model_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        
    def on_model_changed(self, model_size: str):
        """Handle model size change"""
        if self.whisper_model and self.whisper_model.model_loaded:
            self.load_model()
            
    def setup_tabs(self):
        """Setup the transcription tabs"""
        # Clear existing tabs
        self.tab_widget.clear()
        
        # Live transcription tab
        live_tab = LiveTranscriptionTab(self.whisper_model)
        self.tab_widget.addTab(live_tab, "Live Transcription")
        
        # File transcription tab
        file_tab = FileTranscriptionTab(self.whisper_model)
        self.tab_widget.addTab(file_tab, "File Transcription")
        
    def closeEvent(self, event):
        """Handle application close event"""
        # Clean up any running processes
        if hasattr(self, 'tab_widget'):
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                if hasattr(tab, 'stop_recording'):
                    tab.stop_recording()
                    
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 