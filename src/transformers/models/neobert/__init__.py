__all__ = [
    "NeoBERTForMTEB",
    "NeoBERTForSequenceClassification",
    "NeoBERTForTokenClassification",
    "NeoBERTLMHead",
    "NeoBERT",
    "NeoBERTConfig",
    "PosOnlyNeoBERTLMHead",
    "SemOnlyNeoBERTLMHead"
    "softpick",
    "NeoBERTForQuestionAnswering"
    # "NomicBERTForSequenceClassification",
]


from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    # "configuration_neobert": ["NeoBERTConfig"],
    "tokenization_neobert": ["BasicTokenizer", "BertTokenizer", "WordpieceTokenizer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_posbert"] = [
        "NeoBERTConfig",
        "NeoBERTForSequenceClassification",
        "NeoBERTForTokenClassification",
        "NeoBERTLMHead",
        "NeoBERT",
    ]


if TYPE_CHECKING:
    from .modeling_posbert import NeoBERTConfig
    from .tokenization_neobert import BasicTokenizer, BertTokenizer, WordpieceTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_posbert import (
            NeoBERT,
            NeoBERTLMHead,
            NeoBERTForSequenceClassification,
            NeoBERTConfig

        )

from .tokenization_neobert import BasicTokenizer, BertTokenizer, WordpieceTokenizer
from .modeling_posbert import (
    NeoBERTForSequenceClassification,
    NeoBERTForTokenClassification,
    NeoBERTLMHead,
    NeoBERT,
    NeoBERTConfig,
    # NeoBERTForQuestionAnswering
    # NomicBERTForSequenceClassification,
)
