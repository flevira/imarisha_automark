"""Local persistence for structured extraction jobs and teacher decisions.

Legacy extraction dictionaries are not silently converted into structured records.
Use JobRepository.save_page() to atomically retain validated responses and evidence.
"""
from database.structured_repository import JobRepository

__all__ = ['JobRepository']
