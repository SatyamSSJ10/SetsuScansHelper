import deepl

TARGET = "EN"

def translate_japanese(text):
    """
    Translates Japanese text to English using DeepL API.
    """
    return deepl.translate(source_language="JA", target_language=TARGET, text=text)

def translate_chinese(text):
    """
    Translates Chinese text to English using DeepL API.
    """
    return deepl.translate(source_language="ZH", target_language=TARGET, text=text)