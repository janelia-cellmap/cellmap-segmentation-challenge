"""Zip file handling utilities for submissions."""

import logging
import os
import zipfile

from upath import UPath

from .exceptions import ValidationError

MAX_UNCOMPRESSED_SIZE = int(os.getenv("MAX_UNCOMPRESSED_SIZE", 50 * 1024**3))  # 50 GB


def _validate_zip_member(member: zipfile.ZipInfo, target_dir: str) -> None:
    """Validate a single zip member against path traversal and symlink attacks.

    Args:
        member: The zip entry to validate.
        target_dir: The resolved extraction directory.

    Raises:
        ValidationError: If the member is a symlink or would extract outside target_dir.
    """
    # Reject symlinks (external_attr upper 16 bits encode Unix mode; 0o120000 = symlink)
    if member.external_attr >> 16 & 0o170000 == 0o120000:
        raise ValidationError(
            f"Zip member {member.filename!r} is a symlink, which is not allowed."
        )

    # Resolve the destination and ensure it stays within target_dir and not equal to it
    target_real = os.path.realpath(target_dir)
    dest_real = os.path.realpath(os.path.join(target_real, member.filename))
    # Use commonpath to robustly ensure dest_real is a strict descendant of target_real
    if (
        os.path.commonpath([target_real, dest_real]) != target_real
        or dest_real == target_real
    ):
        raise ValidationError(
            f"Zip member {member.filename!r} would extract outside the target directory."
        )


def unzip_file(zip_path, max_uncompressed_size: int = MAX_UNCOMPRESSED_SIZE):
    """Unzip a zip file to a specified directory.

    Validates against path traversal (zip slip), symlink attacks, and
    decompression bombs before extracting.

    Args:
        zip_path (str): The path to the zip file.
        max_uncompressed_size (int): Maximum total uncompressed size in bytes.

    Raises:
        ValidationError: If any member fails security checks or total size exceeds limit.

    Example usage:
        unzip_file('submission.zip')
    """
    logging.info(f"Unzipping {zip_path}...")
    saved_path = UPath(zip_path).with_suffix(".zarr").path
    if UPath(saved_path).exists():
        logging.info(f"Using existing unzipped path at {saved_path}")
        return UPath(saved_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Check total uncompressed size (zip bomb guard)
        total_size = sum(info.file_size for info in zip_ref.infolist())
        if total_size > max_uncompressed_size:
            raise ValidationError(
                f"Zip uncompressed size ({total_size / 1024**3:.1f} GB) exceeds "
                f"limit ({max_uncompressed_size / 1024**3:.1f} GB)."
            )
        saved_path_real = os.path.realpath(saved_path)
        for member in zip_ref.infolist():
            _validate_zip_member(member, saved_path_real)
        for member in zip_ref.infolist():
            _validate_zip_member(member, saved_path)

        zip_ref.extractall(saved_path)

    logging.info(f"Unzipped {zip_path} to {saved_path}")

    return UPath(saved_path)
