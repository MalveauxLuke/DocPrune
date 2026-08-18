"""Version-gated sparse ColPali execution for retrieval indexing."""

from docprune.colpali.compat import (
    ColPaliCompatibility,
    ColPaliCompatibilityError,
    assert_supported_colpali,
)
from docprune.colpali.embedding import ColPaliPageEmbedding, encode_colpali_page
from docprune.colpali.vision import sparse_siglip_features

__all__ = [
    "ColPaliCompatibility",
    "ColPaliCompatibilityError",
    "ColPaliPageEmbedding",
    "assert_supported_colpali",
    "encode_colpali_page",
    "sparse_siglip_features",
]
