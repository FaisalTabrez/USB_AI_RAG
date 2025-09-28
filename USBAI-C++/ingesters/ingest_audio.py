# ingesters/ingest_audio.py
import os
import subprocess
import json
from pydub import AudioSegment
from embeddings.embedder import text_to_embedding
from indexer.faiss_index import add_vector

def convert_to_wav(input_path, output_path=None):
    """Convert audio file to WAV format"""
    try:
        if not output_path:
            output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'

        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error converting audio {input_path}: {e}")
        return None

def run_whisper_cpp(audio_path, model_path="models/ggml-base.bin"):
    """Run whisper.cpp for transcription"""
    try:
        # Check if whisper.cpp binary exists
        whisper_cpp_path = "whisper.cpp/build/bin/main"
        if not os.path.exists(whisper_cpp_path):
            print("whisper.cpp not found, falling back to python whisper")
            return None

        # Run whisper.cpp
        cmd = [
            whisper_cpp_path,
            "-f", audio_path,
            "-m", model_path,
            "-oj",  # JSON output
            "-nt"  # no timestamps
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Parse JSON output
            try:
                output = json.loads(result.stdout)
                return output.get("text", "").strip()
            except:
                return result.stdout.strip()
        else:
            print(f"whisper.cpp failed: {result.stderr}")
            return None

    except FileNotFoundError:
        print("whisper.cpp binary not found")
        return None
    except subprocess.TimeoutExpired:
        print("whisper.cpp timed out")
        return None
    except Exception as e:
        print(f"Error running whisper.cpp: {e}")
        return None

def run_whisper_python(audio_path):
    """Run OpenAI Whisper for transcription (fallback)"""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except ImportError:
        print("OpenAI whisper not available")
        return None
    except Exception as e:
        print(f"Error running whisper: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio file"""
    # First try whisper.cpp
    text = run_whisper_cpp(audio_path)

    # Fallback to python whisper
    if not text:
        text = run_whisper_python(audio_path)

    # Last resort: return filename as text
    if not text:
        text = f"Audio file: {os.path.basename(audio_path)}"

    return text

def get_audio_metadata(audio_path):
    """Get audio file metadata"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return {
            "duration_seconds": len(audio) / 1000,
            "channels": audio.channels,
            "frame_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
            "size_bytes": os.path.getsize(audio_path)
        }
    except Exception as e:
        print(f"Error getting audio metadata: {e}")
        return {}

def chunk_transcript(text, chunk_size=500, overlap=100):
    """Split transcript into chunks"""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

    return chunks

def ingest_audio(audio_path):
    """Ingest an audio file"""
    try:
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False

        print(f"Processing audio: {audio_path}")

        # Get audio metadata
        metadata = get_audio_metadata(audio_path)
        metadata["original_path"] = audio_path

        # Convert to WAV if needed
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext != '.wav':
            wav_path = convert_to_wav(audio_path)
            if not wav_path:
                print(f"Could not convert {audio_path} to WAV")
                return False
        else:
            wav_path = audio_path

        # Transcribe audio
        transcript = transcribe_audio(wav_path)
        print(f"Transcription: {transcript[:100]}...")

        if not transcript:
            print(f"No transcription available for {audio_path}")
            return False

        # Chunk transcript
        chunks = chunk_transcript(transcript)

        # Index each chunk
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Get embedding
            embedding = text_to_embedding(chunk)

            # Prepare metadata for this chunk
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "start_time": chunk_idx * 30,  # Approximate timing
                "end_time": (chunk_idx + 1) * 30,
                "transcript_length": len(transcript)
            }

            # Add to index
            add_vector(
                vector=embedding[0],
                file_path=audio_path,
                file_type="audio",
                content_type="text",
                metadata=chunk_metadata,
                snippet=chunk[:200] + "..." if len(chunk) > 200 else chunk
            )

        print(f"Successfully ingested audio: {audio_path}")
        return True

    except Exception as e:
        print(f"Error ingesting audio {audio_path}: {e}")
        return False

def batch_ingest_audio(directory_path):
    """Ingest all audio files in a directory"""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return False

    success_count = 0
    total_count = 0

    # Supported audio extensions
    supported_exts = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma']

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in supported_exts):
            audio_path = os.path.join(directory_path, filename)
            total_count += 1

            if ingest_audio(audio_path):
                success_count += 1

    print(f"Batch audio ingestion complete: {success_count}/{total_count} files processed")
    return success_count == total_count

def ingest_call_recording(audio_path, call_metadata=None):
    """Ingest a call recording with additional metadata"""
    try:
        metadata = get_audio_metadata(audio_path)
        if call_metadata:
            metadata.update(call_metadata)

        metadata.update({
            "recording_type": "call",
            "call_participants": call_metadata.get("participants", []) if call_metadata else []
        })

        # Convert and transcribe
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext != '.wav':
            wav_path = convert_to_wav(audio_path)
            if not wav_path:
                return False
        else:
            wav_path = audio_path

        transcript = transcribe_audio(wav_path)

        if transcript:
            embedding = text_to_embedding(transcript)

            # Add full transcript
            add_vector(
                vector=embedding[0],
                file_path=audio_path,
                file_type="call_recording",
                content_type="text",
                metadata=metadata,
                snippet=transcript[:200] + "..." if len(transcript) > 200 else transcript
            )

        return True

    except Exception as e:
        print(f"Error ingesting call recording {audio_path}: {e}")
        return False

if __name__ == "__main__":
    # Test the ingester
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            batch_ingest_audio(path)
        else:
            ingest_audio(path)
    else:
        print("Usage: python ingest_audio.py <audio_path_or_directory>")
