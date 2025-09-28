#!/usr/bin/env python3
"""
Demo setup script for Multimodal RAG System
This script helps users get started quickly with sample data and configuration.
"""

import os
import sys
import shutil
from pathlib import Path

def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")

    # Create sample documents directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create sample PDF content (simulated)
    sample_pdf_dir = data_dir / "reports"
    sample_pdf_dir.mkdir(exist_ok=True)

    # Create sample images directory
    sample_images_dir = data_dir / "screenshots"
    sample_images_dir.mkdir(exist_ok=True)

    # Create sample audio directory
    sample_audio_dir = data_dir / "recordings"
    sample_audio_dir.mkdir(exist_ok=True)

    # Create sample documents
    create_sample_documents(sample_pdf_dir)
    create_sample_images(sample_images_dir)
    create_sample_audio(sample_audio_dir)

    print(f"Sample data created in: {data_dir}")
    return True

def create_sample_documents(docs_dir):
    """Create sample document files"""
    # Annual Report (simulated as text file)
    report_content = """
    ANNUAL REPORT 2024

    EXECUTIVE SUMMARY

    This year has been exceptional for our organization. We achieved record-breaking
    revenue of $2.5 million, representing a 25% increase from the previous year.

    KEY ACHIEVEMENTS:
    - Revenue Growth: 25% year-over-year
    - Market Expansion: Entered 3 new international markets
    - Customer Satisfaction: 95% satisfaction rate
    - Employee Growth: 40 new hires across all departments

    FINANCIAL HIGHLIGHTS:
    - Total Revenue: $2,500,000
    - Net Profit: $380,000
    - R&D Investment: $150,000
    - Marketing Budget: $200,000

    STRATEGIC INITIATIVES:
    1. Digital Transformation: Implemented new CRM system
    2. Sustainability: Reduced carbon footprint by 30%
    3. Innovation: Launched 5 new products
    4. Global Expansion: Opened offices in Europe and Asia

    CONCLUSION:
    The coming year looks promising with continued growth expected at 20%.
    Our focus will remain on innovation, customer satisfaction, and sustainable practices.

    Prepared by: John Smith, CEO
    Date: December 15, 2024
    """

    with open(docs_dir / "annual_report_2024.txt", 'w') as f:
        f.write(report_content)

    # Meeting Notes
    meeting_content = """
    WEEKLY STANDUP MEETING - MARCH 15, 2024

    ATTENDEES:
    - Sarah Johnson (Project Manager)
    - Mike Chen (Lead Developer)
    - Alex Rodriguez (Designer)
    - Emily Davis (QA Engineer)

    DISCUSSION POINTS:

    1. PROJECT DEADLINES
    - Q2 release moved from April 30 to May 15
    - Resource constraints in development team
    - Sarah to discuss with external contractors

    2. NEW FEATURES
    - User authentication module completed
    - Dashboard redesign approved
    - Mobile responsiveness improvements needed

    3. BUG REPORTS
    - Login timeout issue resolved
    - Performance optimization completed
    - Two minor UI bugs identified

    4. NEXT WEEK'S PRIORITIES
    - Complete API integration testing
    - Finalize mobile design mockups
    - Security audit preparation

    ACTION ITEMS:
    - Sarah: Contact contractors by Friday
    - Mike: Schedule security audit
    - Alex: Prepare mobile mockups
    - Emily: Complete API testing

    Next meeting: March 22, 2024 at 10:00 AM
    """

    with open(docs_dir / "meeting_notes_2024-03-15.txt", 'w') as f:
        f.write(meeting_content)

def create_sample_images(images_dir):
    """Create sample image metadata (placeholder)"""
    # Create a text file representing an image with OCR content
    screenshot_content = """
    DASHBOARD SCREENSHOT - 14:32

    KPI DASHBOARD - REAL TIME METRICS

    REVENUE: $2,500,000 + 25%
    ACTIVE USERS: 15,420 + 12%
    CONVERSION RATE: 3.2% + 0.5%
    CUSTOMER SATISFACTION: 95% + 2%

    RECENT ACTIVITY:
    - New user registration spike at 14:15
    - Server response time: 120ms
    - Database queries: 1,247/min
    - Error rate: 0.02%

    SYSTEM STATUS: ALL SYSTEMS OPERATIONAL
    LAST UPDATED: 14:32:15
    """

    with open(images_dir / "dashboard_screenshot_14_32.txt", 'w') as f:
        f.write(screenshot_content)

def create_sample_audio(audio_dir):
    """Create sample audio transcript"""
    transcript_content = """
    TEAM MEETING RECORDING - MARCH 10, 2024

    [00:00:00 - 00:02:30] Introduction and agenda review

    SARAH: Good morning everyone. Today we'll discuss the Q2 project timeline,
    resource allocation, and upcoming deadlines. Let's start with project status updates.

    [00:02:30 - 00:05:20] Development team update

    MIKE: The development team has completed the core authentication module.
    We're currently working on the dashboard redesign and expect to finish by next Friday.
    However, we're running behind on the mobile responsiveness features.

    [00:05:20 - 00:08:15] Deadline discussion

    SARAH: Mike, you mentioned being behind on mobile features. How much time do you need?

    MIKE: I'd say we need at least two additional weeks. The current deadline of April 30th
    is too aggressive given our current resource constraints.

    ALEX: I agree with Mike. The design complexity for mobile is higher than initially estimated.

    SARAH: Let me check with management about extending the deadline. We might need to bring in
    external contractors to help meet the original timeline.

    [00:08:15 - 00:10:45] Quality assurance update

    EMILY: QA has completed testing on the authentication module. We found and fixed two critical bugs.
    Performance testing shows improved response times. We're ready to start API integration testing.

    [00:10:45 - 00:12:00] Action items and next steps

    SARAH: Alright, let's summarize action items:
    1. Mike to provide detailed mobile development estimate by Wednesday
    2. I'll discuss deadline extension with stakeholders
    3. Emily to begin API integration testing
    4. Alex to finalize mobile design requirements

    Next meeting scheduled for March 17th.
    """

    with open(audio_dir / "team_meeting_2024-03-10.txt", 'w') as f:
        f.write(transcript_content)

def download_sample_model():
    """Download a small sample model for demonstration"""
    print("Setting up sample model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("Note: For demo purposes, please manually download a small GGML model")
    print("Recommended: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF")
    print("Place the model file in the models/ directory")

def create_config_file():
    """Create a sample configuration file"""
    config_content = """# Multimodal RAG Configuration
[data]
data_directory = data
database_directory = db
models_directory = models

[processing]
chunk_size = 800
overlap = 150
max_results = 5

[model]
model_path = models/your_model.gguf
temperature = 0.1
max_tokens = 512

[ui]
web_port = 5000
debug_mode = true
"""

    with open("config.ini", 'w') as f:
        f.write(config_content)

    print("Configuration file created: config.ini")

def main():
    """Main setup function"""
    print("🤖 Multimodal RAG System - Demo Setup")
    print("=" * 50)

    # Create sample data
    if create_sample_data():
        print("✅ Sample data created successfully")
    else:
        print("❌ Failed to create sample data")
        return False

    # Setup model (instructions)
    download_sample_model()

    # Create config file
    create_config_file()

    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Download a GGML model to the models/ directory")
    print("2. Install dependencies: python main.py setup")
    print("3. Ingest sample data: python main.py ingest data")
    print("4. Start web interface: python webapp/app.py")
    print("5. Or try CLI: python main.py query 'What are the key findings?'")

    print("\nSample questions to try:")
    print("- 'What are the key achievements in the annual report?'")
    print("- 'What was discussed about project deadlines?'")
    print("- 'Show me the dashboard screenshot from 14:32'")
    print("- 'What are the action items from the team meeting?'")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
