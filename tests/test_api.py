import unittest
from pathlib import Path

try:
    from fastapi import HTTPException
    from api.main import validate_session_id
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TestApiSyntax(unittest.TestCase):

    def test_api_module_compiles_without_optional_dependencies(self):
        source = Path("api/main.py").read_text(encoding="utf-8")
        compile(source, "api/main.py", "exec")


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI API extras not installed")
class TestApiValidation(unittest.TestCase):

    def test_accepts_safe_session_id(self):
        self.assertEqual(validate_session_id("session-01_user"), "session-01_user")

    def test_rejects_unsafe_session_id(self):
        with self.assertRaises(HTTPException) as error:
            validate_session_id("session:other")
        self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
