"""S3 integration service."""

import io
import json
from typing import Optional, BinaryIO
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import get_settings


class S3Service:
    """Service for S3 operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    @property
    def client(self):
        """Lazy-load S3 client."""
        if self._client is None:
            try:
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=self.settings.aws_access_key_id or None,
                    aws_secret_access_key=self.settings.aws_secret_access_key or None,
                    region_name=self.settings.aws_region,
                )
            except NoCredentialsError:
                # Return None to indicate S3 is not available
                return None
        return self._client
    
    @property
    def bucket(self) -> str:
        return self.settings.s3_bucket
    
    def is_available(self) -> bool:
        """Check if S3 is available and configured."""
        if self.client is None:
            return False
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False
        except Exception:
            return False
    
    def upload_file(
        self, 
        file: BinaryIO, 
        key: str, 
        content_type: str = "text/csv"
    ) -> bool:
        """Upload a file to S3."""
        if self.client is None:
            return False
        try:
            self.client.upload_fileobj(
                file,
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type}
            )
            return True
        except ClientError as e:
            print(f"S3 upload error: {e}")
            return False
    
    def download_file(self, key: str) -> Optional[bytes]:
        """Download a file from S3."""
        if self.client is None:
            return None
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except ClientError:
            return None
    
    def upload_json(self, data: dict, key: str) -> bool:
        """Upload JSON data to S3."""
        if self.client is None:
            return False
        try:
            json_bytes = json.dumps(data).encode('utf-8')
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_bytes,
                ContentType='application/json'
            )
            return True
        except ClientError:
            return False
    
    def download_json(self, key: str) -> Optional[dict]:
        """Download JSON data from S3."""
        content = self.download_file(key)
        if content:
            return json.loads(content.decode('utf-8'))
        return None
    
    def list_objects(self, prefix: str = "") -> list[str]:
        """List objects in S3 with given prefix."""
        if self.client is None:
            return []
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError:
            return []
    
    def delete_object(self, key: str) -> bool:
        """Delete an object from S3."""
        if self.client is None:
            return False
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


# Singleton instance
s3_service = S3Service()
