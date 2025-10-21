from google.cloud import vision
import os

# Check if credentials are set
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if creds_path:
    print(f"✓ Credentials path: {creds_path}")
    print(f"✓ File exists: {os.path.exists(creds_path)}")
    
    # Try to create a client
    try:
        client = vision.ImageAnnotatorClient()
        print("✓ Google Cloud Vision API client created successfully!")
    except Exception as e:
        print(f"✗ Error creating client: {e}")
else:
    print("✗ GOOGLE_APPLICATION_CREDENTIALS not set")