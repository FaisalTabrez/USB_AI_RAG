# rag/prompt_builder.py
import os
from datetime import datetime

PROMPT_HEADER = """You are a helpful assistant that must **only** use the provided sources (evidence) to answer.
Do NOT hallucinate. If the answer is not contained in the evidence, say "I don't know" or ask for clarification.
Cite every claim using the evidence numbers in brackets, e.g. [1], [2].
At the end, provide a "SOURCES" list mapping each evidence number to the original file path and offsets.

IMPORTANT RULES:
1. Only use information from the provided evidence blocks
2. Cite every statement with evidence numbers like [1], [2]
3. If evidence is insufficient, say "I don't know - the provided sources don't contain the answer"
4. Include exact file paths and metadata in the SOURCES section
5. Be concise but comprehensive"""

def format_evidence_block(evidence, index):
    """Format a single evidence block with proper metadata"""
    doc = evidence.get('document', {})
    file_path = doc.get('file_path', 'Unknown')
    file_type = doc.get('file_type', 'Unknown')
    metadata = doc.get('metadata', {})

    # Build location string
    location_parts = []

    if file_type == 'pdf' and 'page' in metadata:
        location_parts.append(f"page {metadata['page']}")

    if file_type in ['audio', 'call_recording'] and 'start_time' in metadata:
        start_time = metadata.get('start_time', 0)
        end_time = metadata.get('end_time', 0)
        location_parts.append(f"{start_time:.1f}s-{end_time:.1f}s")

    if file_type == 'image' and 'width' in metadata:
        location_parts.append(f"{metadata['width']}x{metadata['height']}")

    location = f" ({', '.join(location_parts)})" if location_parts else ""

    # Get snippet
    snippet = evidence.get('snippet', '')

    return f"[{index}] {file_type.upper()}: {os.path.basename(file_path)}{location}\n---\n{snippet}\n"

def build_prompt(user_query, evidence_list, max_chars=3000):
    """
    Build a RAG prompt with numbered evidence and strict instructions

    Args:
        user_query: The user's question
        evidence_list: List of evidence dictionaries from FAISS search
        max_chars: Maximum characters for evidence text
    """
    if not evidence_list:
        return f"{PROMPT_HEADER}\n\nNo evidence found for query.\n\nUser question: {user_query}\n\nPlease say 'I don't know' or ask for clarification."

    # Format evidence blocks
    evidence_blocks = []
    citation_map = []

    for i, evidence in enumerate(evidence_list[:10], start=1):  # Limit to top 10 results
        try:
            block = format_evidence_block(evidence, i)
            evidence_blocks.append(block)

            # Build citation mapping
            doc = evidence.get('document', {})
            file_path = doc.get('file_path', 'Unknown')
            metadata = doc.get('metadata', {})

            location = ""
            if 'page' in metadata:
                location += f"page {metadata['page']}"
            if 'start_time' in metadata:
                location += f" {metadata['start_time']:.1f}s-{metadata['end_time']:.1f}s"

            citation_map.append(f"[{i}] {file_path}{f' ({location})' if location else ''}")

        except Exception as e:
            print(f"Error formatting evidence {i}: {e}")
            continue

    if not evidence_blocks:
        return f"{PROMPT_HEADER}\n\nError processing evidence.\n\nUser question: {user_query}\n\nPlease say 'I don't know'."

    # Combine evidence text
    combined_evidence = "\n\n".join(evidence_blocks)

    # Truncate if too long
    if len(combined_evidence) > max_chars:
        combined_evidence = combined_evidence[:max_chars] + "\n... [truncated]"

    # Build sources section
    sources_text = "\n".join(citation_map)

    # Complete prompt
    prompt = f"""{PROMPT_HEADER}

EVIDENCE:
{combined_evidence}

User question: {user_query}

Answer concisely, grounding every statement in the evidence above with bracket citations like [1], [2].
After the answer, include:

SOURCES:
{sources_text}
"""

    return prompt

def build_multimodal_prompt(user_query, text_evidence, image_evidence=None, audio_evidence=None):
    """
    Build a prompt for multimodal queries with different evidence types
    """
    all_evidence = []

    # Add text evidence
    for i, evidence in enumerate(text_evidence[:5]):
        all_evidence.append({
            'document': evidence.get('document', {}),
            'snippet': evidence.get('snippet', ''),
            'type': 'text'
        })

    # Add image evidence
    if image_evidence:
        for i, evidence in enumerate(image_evidence[:3]):
            doc = evidence.get('document', {})
            metadata = doc.get('metadata', {})
            all_evidence.append({
                'document': doc,
                'snippet': f"Image ({metadata.get('width', 0)}x{metadata.get('height', 0)}) - OCR: {evidence.get('snippet', '')}",
                'type': 'image'
            })

    # Add audio evidence
    if audio_evidence:
        for i, evidence in enumerate(audio_evidence[:3]):
            doc = evidence.get('document', {})
            metadata = doc.get('metadata', {})
            all_evidence.append({
                'document': doc,
                'snippet': f"Audio transcript: {evidence.get('snippet', '')}",
                'type': 'audio'
            })

    return build_prompt(user_query, all_evidence)

def create_citation_summary(results):
    """Create a summary of citations for the response"""
    citations = []

    for i, result in enumerate(results[:5], start=1):
        doc = result.get('document', {})
        file_path = doc.get('file_path', 'Unknown')
        file_type = doc.get('file_type', 'Unknown')
        metadata = doc.get('metadata', {})

        citation = f"[{i}] {file_type.upper()}: {os.path.basename(file_path)}"

        if 'page' in metadata:
            citation += f" (page {metadata['page']})"
        elif 'start_time' in metadata:
            citation += f" ({metadata['start_time']:.1f}s)"

        citations.append(citation)

    return citations
