import unittest

import tools


class RuntimeRequirementsTest(unittest.TestCase):
    def test_require_python_accepts_supported_version(self):
        tools.require_python(version_info=(3, 12, 0))

    def test_require_python_rejects_older_version(self):
        with self.assertRaisesRegex(RuntimeError, "Python >= 3.12 is required"):
            tools.require_python(version_info=(3, 10, 22))


if __name__ == "__main__":
    unittest.main()
