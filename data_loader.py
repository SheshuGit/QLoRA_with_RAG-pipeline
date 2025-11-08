# data_loader.py
import boto3
from pathlib import Path
from config import Config
import os

class S3DataLoader:
    
    def __init__(self):
        # Check if S3 credentials are configured
        if not Config.AWS_ACCESS_KEY or not Config.AWS_SECRET_KEY:
            print("⚠ Warning: AWS credentials not configured in .env file")
            print("  S3 download will not work. Please configure:")
            print("  - AWS_ACCESS_KEY")
            print("  - AWS_SECRET_KEY")
            print("  - S3_BUCKET")
            self.s3 = None
            self.bucket = None
            return
        
        if not Config.S3_BUCKET:
            print("⚠ Warning: S3_BUCKET not configured in .env file")
            self.s3 = None
            self.bucket = None
            return
            
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=Config.AWS_ACCESS_KEY,
                aws_secret_access_key=Config.AWS_SECRET_KEY
            )
            self.bucket = Config.S3_BUCKET
        except Exception as e:
            print(f"⚠ Error initializing S3 client: {e}")
            self.s3 = None
            self.bucket = None
        
    def list_files(self, prefix=''):
        """List all files in S3 bucket with given prefix"""
        if self.s3 is None or self.bucket is None:
            print("✗ S3 client not initialized. Cannot list files.")
            return []
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"✗ Error listing S3 files: {e}")
            return []
    
    def download_file(self, s3_key, local_path):
        """Download file from S3 to local path"""
        if self.s3 is None or self.bucket is None:
            print("✗ S3 client not initialized. Cannot download files.")
            return None
            
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_key, local_path)
            return local_path
        except Exception as e:
            print(f"✗ Error downloading {s3_key}: {e}")
            return None
    
    def load_documents(self, local_dir):
        """Load all documents from S3 to local directory"""
        if self.s3 is None or self.bucket is None:
            print("✗ S3 not configured. Skipping download.")
            print("  Please configure AWS credentials in .env file")
            return []
            
        files = self.list_files()
        if not files:
            print("⚠ No files found in S3 bucket")
            return []
            
        print(f"Found {len(files)} files in S3 bucket")
        downloaded = []
        
        # Ensure we download to raw subdirectory
        raw_dir = os.path.join(local_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        for file_key in files:
            # Get just the filename without S3 path
            filename = os.path.basename(file_key)
            local_path = os.path.join(raw_dir, filename)
            result = self.download_file(file_key, local_path)
            if result:
                downloaded.append(result)
                print(f"  ✓ Downloaded: {filename} → data/raw/")
        
        print(f"✓ Downloaded {len(downloaded)} files to data/raw/")
        return [str(p) for p in Path(raw_dir).rglob('*') if p.is_file()]

if __name__ == "__main__":
    print("🚀 Starting S3 data loading test...")
    loader = S3DataLoader()

    print("🧾 Bucket:", loader.bucket)
    print("🧾 S3 client initialized:", bool(loader.s3))

    # Only proceed if S3 client is ready
    if loader.s3 and loader.bucket:
        print("🔍 Listing files in S3 bucket...")
        files = loader.list_files()
        print(f"📂 Files found: {files}")

        # Test download if files exist
        if files:
            local_dir = "data"
            print(f"⬇️ Downloading first file: {files[0]}")
            loader.download_file(files[0], os.path.join(local_dir, "raw", os.path.basename(files[0])))
        else:
            print("⚠️ No files found in the bucket.")
    else:
        print("❌ S3 client not initialized. Please check your .env configuration.")
