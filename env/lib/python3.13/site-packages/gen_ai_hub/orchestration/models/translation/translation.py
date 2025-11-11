from gen_ai_hub.orchestration.models.base import JSONSerializable
from enum import Enum


class TranslationType(str, Enum):
    """Enumerates supported translation types."""
    SAP_DOCUMENT_TRANSLATION = "sap_document_translation"


class InputTranslationConfig(JSONSerializable):
    """Configuration for input translation. These parameters are specific to SAP Translation Hub.

    Args:
        source_language: The source language code (e.g., 'de-DE' for German).
        target_language: The target language code (e.g., 'en-US' for US English).
    """

    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language

    def to_dict(self):
        return {
            "source_language": self.source_language,
            "target_language": self.target_language
        }


class InputTranslationModule(JSONSerializable):
    """Configuration for input translation module.

    Args:
        type: The type of translation module (e.g., 'sap_document_translation').
        config: Configuration object for the translation module.
    """

    def __init__(self, type: str, config: InputTranslationConfig):
        self.type = type
        self.config = config

    def to_dict(self):
        return {
            "type": self.type,
            "config": self.config.to_dict()
        }


class OutputTranslationConfig(JSONSerializable):
    """Configuration for output translation. These parameters are specific to SAP Translation Hub.

    Args:
        source_language: The source language code (e.g., 'de-DE' for German).
        target_language: The target language code (e.g., 'en-US' for US English).
    """

    def __init__(self, target_language: str, source_language: str = None):
        self.target_language = target_language
        self.source_language = source_language

    def to_dict(self):
        return {
            "target_language": self.target_language,
            "source_language": self.source_language
        }


class OutputTranslationModule(JSONSerializable):
    """Configuration for output translation module.

    Args:
        type: The type of translation module (e.g., 'sap_document_translation').
        config: Configuration object for the translation module.
    """

    def __init__(self, type: str, config: OutputTranslationConfig):
        self.type = type
        self.config = config

    def to_dict(self):
        return {
            "type": self.type,
            "config": self.config.to_dict()
        }


class Translation:
    """
    Translation module for managing input and output translations.

    Args:
        input_translation: Configuration object for input translation.
        output_translation: Configuration object for output translation.
    """

    def __init__(self, input_translation: InputTranslationModule = None,
                 output_translation: OutputTranslationModule = None):
        self.input_translation = input_translation
        self.output_translation = output_translation
