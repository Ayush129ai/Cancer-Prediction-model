import pytest
import sys
import codecs

if __name__ == "__main__":
    with codecs.open('pytest_output_clean.txt', 'w', 'utf-8') as f:
        class RedirectStdout:
            def write(self, s):
                f.write(s)
            def flush(self):
                f.flush()
        sys.stdout = RedirectStdout()
        sys.stderr = RedirectStdout()
        pytest.main(['tests'])
