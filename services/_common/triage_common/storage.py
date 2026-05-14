from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import BinaryIO, Iterator, Optional

from minio import Minio


BUCKET_AUDIO_ORIGINAL = "audio-original"
BUCKET_TEXTOS_ORIGINALES = "textos-originales"
BUCKET_DATASETS = "datasets"
BUCKET_MODELOS = "modelos"

ALL_BUCKETS = (
    BUCKET_AUDIO_ORIGINAL,
    BUCKET_TEXTOS_ORIGINALES,
    BUCKET_DATASETS,
    BUCKET_MODELOS,
)


@dataclass(frozen=True)
class StorageConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: Optional[str] = None

    @classmethod
    def from_env(cls) -> "StorageConfig":
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ROOT_USER", "minio_admin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minio_admin_pw"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            region=os.getenv("MINIO_REGION") or None,
        )


class StorageClient:
    def __init__(self, config: StorageConfig | None = None, client: Optional[Minio] = None) -> None:
        self._config = config or StorageConfig.from_env()
        self._client = client or Minio(
            self._config.endpoint,
            access_key=self._config.access_key,
            secret_key=self._config.secret_key,
            secure=self._config.secure,
            region=self._config.region,
        )

    @property
    def raw(self) -> Minio:
        return self._client

    def put_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        return self._uri(bucket, object_name)

    def put_stream(
        self,
        bucket: str,
        object_name: str,
        stream: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        self._client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=stream,
            length=length,
            content_type=content_type,
        )
        return self._uri(bucket, object_name)

    def put_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        self._client.fput_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type,
        )
        return self._uri(bucket, object_name)

    def get_bytes(self, bucket: str, object_name: str) -> bytes:
        response = self._client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_to_file(self, bucket: str, object_name: str, file_path: str) -> None:
        self._client.fget_object(bucket, object_name, file_path)

    def presign_url(
        self,
        bucket: str,
        object_name: str,
        expires_seconds: int = 3600,
    ) -> str:
        return self._client.presigned_get_object(
            bucket, object_name, expires=timedelta(seconds=expires_seconds)
        )

    def list_objects(
        self, bucket: str, prefix: Optional[str] = None
    ) -> Iterator[str]:
        for obj in self._client.list_objects(bucket, prefix=prefix, recursive=True):
            yield obj.object_name

    def ensure_buckets(self, buckets: tuple[str, ...] = ALL_BUCKETS) -> None:
        for bucket in buckets:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)

    @staticmethod
    def _uri(bucket: str, object_name: str) -> str:
        return f"s3://{bucket}/{object_name}"


def parse_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"URI no soportada: {uri}")
    body = uri[len("s3://"):]
    bucket, _, object_name = body.partition("/")
    if not bucket or not object_name:
        raise ValueError(f"URI mal formada: {uri}")
    return bucket, object_name
