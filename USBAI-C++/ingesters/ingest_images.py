# ingesters/ingest_images.py
import os
from PIL import Image
import pytesseract
from embeddings.embedder import image_to_embedding, text_to_embedding
from indexer.faiss_index import add_vector

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    try:
        img = Image.open(image_path)
        # Convert to grayscale for better OCR results
        if img.mode != 'L':
            img = img.convert('L')

        # Perform OCR
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error during OCR on {image_path}: {e}")
        return ""

def get_image_metadata(image_path):
    """Get basic image metadata"""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "size_bytes": os.path.getsize(image_path)
            }
    except Exception as e:
        print(f"Error getting metadata for {image_path}: {e}")
        return {}

def ingest_image(image_path):
    """Ingest a single image file"""
    try:
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return False

        print(f"Processing image: {image_path}")

        # Get image metadata
        metadata = get_image_metadata(image_path)
        metadata["original_path"] = image_path

        # Extract text using OCR
        ocr_text = extract_text_from_image(image_path)

        # Get image embedding
        image_embedding = image_to_embedding(image_path)

        # Prepare metadata for image vector
        image_metadata = {
            **metadata,
            "content_type": "image",
            "has_ocr_text": bool(ocr_text.strip())
        }

        # Add image embedding to index
        add_vector(
            vector=image_embedding[0],
            file_path=image_path,
            file_type="image",
            content_type="image",
            metadata=image_metadata,
            snippet=f"Image ({metadata.get('width', 0)}x{metadata.get('height', 0)})"
        )

        # If OCR text is substantial, also index it as text
        if ocr_text and len(ocr_text.strip()) > 20:
            text_embedding = text_to_embedding(ocr_text)

            text_metadata = {
                **metadata,
                "content_type": "ocr_text",
                "source_image": image_path
            }

            add_vector(
                vector=text_embedding[0],
                file_path=image_path,
                file_type="image",
                content_type="text",
                metadata=text_metadata,
                snippet=ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
            )

        print(f"Successfully ingested image: {image_path}")
        return True

    except Exception as e:
        print(f"Error ingesting image {image_path}: {e}")
        return False

def batch_ingest_images(directory_path):
    """Ingest all images in a directory"""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return False

    success_count = 0
    total_count = 0

    # Supported image extensions
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']

    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in supported_exts):
            image_path = os.path.join(directory_path, filename)
            total_count += 1

            if ingest_image(image_path):
                success_count += 1

    print(f"Batch image ingestion complete: {success_count}/{total_count} images processed")
    return success_count == total_count

def ingest_screenshot(image_path, timestamp=None, window_title=None):
    """Ingest a screenshot with additional metadata"""
    try:
        metadata = get_image_metadata(image_path)
        metadata.update({
            "screenshot": True,
            "timestamp": timestamp,
            "window_title": window_title
        })

        # Get embedding
        embedding = image_to_embedding(image_path)

        # Add to index
        add_vector(
            vector=embedding[0],
            file_path=image_path,
            file_type="screenshot",
            content_type="image",
            metadata=metadata,
            snippet=f"Screenshot at {timestamp}: {window_title or 'Unknown window'}"
        )

        # Extract and index any text in the screenshot
        ocr_text = extract_text_from_image(image_path)
        if ocr_text and len(ocr_text.strip()) > 10:
            text_embedding = text_to_embedding(ocr_text)

            text_metadata = {
                **metadata,
                "content_type": "screenshot_text"
            }

            add_vector(
                vector=text_embedding[0],
                file_path=image_path,
                file_type="screenshot",
                content_type="text",
                metadata=text_metadata,
                snippet=ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
            )

        return True

    except Exception as e:
        print(f"Error ingesting screenshot {image_path}: {e}")
        return False

if __name__ == "__main__":
    # Test the ingester
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            batch_ingest_images(path)
        else:
            ingest_image(path)
    else:
        print("Usage: python ingest_images.py <image_path_or_directory>")
