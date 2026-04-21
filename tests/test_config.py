import unittest

from llmproxy.config import validate_backends


def _valid():
    return {"m": {"url": "x", "token": "t", "device": "d",
        "model": "org/m", "type": "chat"}}


class TestValidateBackends(unittest.TestCase):
    def test_ok_chat_without_context_length(self):
        validate_backends(_valid())

    def test_ok_with_context_length(self):
        b = _valid(); b["m"]["context_length"] = 4096
        validate_backends(b)

    def test_ok_audio(self):
        b = _valid(); b["m"]["type"] = "audio"
        validate_backends(b)

    def test_ok_with_model_url(self):
        b = _valid()
        b["m"]["model_url"] = "https://example.com/model"
        validate_backends(b)

    def test_missing_model(self):
        b = _valid(); del b["m"]["model"]
        with self.assertRaisesRegex(ValueError, "model"):
            validate_backends(b)

    def test_missing_url(self):
        b = _valid(); del b["m"]["url"]
        with self.assertRaisesRegex(ValueError, "url"):
            validate_backends(b)

    def test_missing_token(self):
        b = _valid(); del b["m"]["token"]
        with self.assertRaisesRegex(ValueError, "token"):
            validate_backends(b)

    def test_missing_device(self):
        b = _valid(); del b["m"]["device"]
        with self.assertRaisesRegex(ValueError, "device"):
            validate_backends(b)

    def test_bad_url_type(self):
        b = _valid(); b["m"]["url"] = 123
        with self.assertRaisesRegex(ValueError, "url must be str"):
            validate_backends(b)

    def test_bad_url_empty(self):
        b = _valid(); b["m"]["url"] = ""
        with self.assertRaisesRegex(ValueError, "url must be non-empty"):
            validate_backends(b)

    def test_bad_url_whitespace(self):
        b = _valid(); b["m"]["url"] = "http://x "
        with self.assertRaisesRegex(ValueError, "url must be non-empty"):
            validate_backends(b)

    def test_missing_type(self):
        b = _valid(); del b["m"]["type"]
        with self.assertRaisesRegex(ValueError, "type"):
            validate_backends(b)

    def test_bad_type(self):
        b = _valid(); b["m"]["type"] = "video"
        with self.assertRaisesRegex(ValueError, "type"):
            validate_backends(b)

    def test_bad_context_length_type(self):
        b = _valid(); b["m"]["context_length"] = "4k"
        with self.assertRaisesRegex(ValueError, "context_length"):
            validate_backends(b)

    def test_context_length_rejected_for_audio(self):
        b = _valid()
        b["m"]["type"] = "audio"
        b["m"]["context_length"] = 4096
        with self.assertRaisesRegex(ValueError,
                "context_length not applicable for type=audio"):
            validate_backends(b)

    def test_ok_with_quantization(self):
        b = _valid(); b["m"]["quantization"] = "FP8"
        validate_backends(b)

    def test_bad_quantization_type(self):
        b = _valid(); b["m"]["quantization"] = 8
        with self.assertRaisesRegex(ValueError, "quantization must be str"):
            validate_backends(b)

    def test_bad_quantization_empty(self):
        b = _valid(); b["m"]["quantization"] = ""
        with self.assertRaisesRegex(ValueError, "quantization must be non-empty"):
            validate_backends(b)

    def test_bad_quantization_whitespace(self):
        b = _valid(); b["m"]["quantization"] = "FP8 "
        with self.assertRaisesRegex(ValueError, "quantization must be non-empty"):
            validate_backends(b)

    def test_accumulates_errors(self):
        b = {"a": {}, "b": {"url": "x"}}
        with self.assertRaises(ValueError) as cm:
            validate_backends(b)
        msg = str(cm.exception)
        self.assertIn("'a'", msg)
        self.assertIn("'b'", msg)
