from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from triage_common import storage
from triage_common.storage import (
    ALL_BUCKETS,
    BUCKET_AUDIO_ORIGINAL,
    BUCKET_DATASETS,
    StorageClient,
    parse_uri,
)


@pytest.fixture
def mock_minio():
    return MagicMock()


@pytest.fixture
def client(mock_minio):
    return StorageClient(client=mock_minio)


class TestPutBytes:
    def test_writes_bytes_and_returns_uri(self, client, mock_minio):
        uri = client.put_bytes(BUCKET_AUDIO_ORIGINAL, "g1.wav", b"hello", "audio/wav")
        assert uri == "s3://audio-original/g1.wav"
        mock_minio.put_object.assert_called_once()
        kwargs = mock_minio.put_object.call_args.kwargs
        assert kwargs["bucket_name"] == BUCKET_AUDIO_ORIGINAL
        assert kwargs["object_name"] == "g1.wav"
        assert kwargs["length"] == len(b"hello")
        assert kwargs["content_type"] == "audio/wav"


class TestPutFile:
    def test_calls_fput(self, client, mock_minio):
        uri = client.put_file(BUCKET_DATASETS, "batch.parquet", "/tmp/x.parquet")
        assert uri == "s3://datasets/batch.parquet"
        mock_minio.fput_object.assert_called_once_with(
            bucket_name=BUCKET_DATASETS,
            object_name="batch.parquet",
            file_path="/tmp/x.parquet",
            content_type="application/octet-stream",
        )


class TestGetBytes:
    def test_reads_and_closes(self, client, mock_minio):
        response = MagicMock()
        response.read.return_value = b"payload"
        mock_minio.get_object.return_value = response
        data = client.get_bytes(BUCKET_AUDIO_ORIGINAL, "g1.wav")
        assert data == b"payload"
        response.close.assert_called_once()
        response.release_conn.assert_called_once()


class TestPresignUrl:
    def test_uses_expires_seconds(self, client, mock_minio):
        mock_minio.presigned_get_object.return_value = "https://signed"
        url = client.presign_url(BUCKET_AUDIO_ORIGINAL, "g1.wav", expires_seconds=600)
        assert url == "https://signed"
        kwargs = mock_minio.presigned_get_object.call_args
        assert kwargs.args[0] == BUCKET_AUDIO_ORIGINAL
        assert kwargs.args[1] == "g1.wav"
        assert kwargs.kwargs["expires"].total_seconds() == 600


class TestEnsureBuckets:
    def test_creates_missing_buckets(self, client, mock_minio):
        mock_minio.bucket_exists.side_effect = lambda b: b == BUCKET_AUDIO_ORIGINAL
        client.ensure_buckets()
        created = [c.args[0] for c in mock_minio.make_bucket.call_args_list]
        assert set(created) == set(ALL_BUCKETS) - {BUCKET_AUDIO_ORIGINAL}


class TestParseUri:
    def test_valid(self):
        assert parse_uri("s3://datasets/x.parquet") == ("datasets", "x.parquet")

    def test_nested_path(self):
        assert parse_uri("s3://audio-original/2026/05/g1.wav") == (
            "audio-original",
            "2026/05/g1.wav",
        )

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            parse_uri("http://x")

    def test_missing_object(self):
        with pytest.raises(ValueError):
            parse_uri("s3://audio-original/")


class TestListObjects:
    def test_yields_object_names(self, client, mock_minio):
        obj1 = MagicMock(); obj1.object_name = "a.txt"
        obj2 = MagicMock(); obj2.object_name = "b.txt"
        mock_minio.list_objects.return_value = iter([obj1, obj2])
        names = list(client.list_objects(BUCKET_DATASETS, prefix="2026/"))
        assert names == ["a.txt", "b.txt"]
        mock_minio.list_objects.assert_called_once_with(
            BUCKET_DATASETS, prefix="2026/", recursive=True
        )
