import unittest

from src.build_features import extract_features


class URLFeatureTests(unittest.TestCase):
    def test_https_url_with_query(self):
        features = extract_features("https://example.com/a?x=12")
        self.assertEqual(features["uses_https"], 1)
        self.assertEqual(features["hostname_length"], 11)
        self.assertEqual(features["path_length"], 2)
        self.assertEqual(features["digit_count"], 2)
        self.assertEqual(features["hostname_is_ip"], 0)

    def test_ip_address_hostname(self):
        features = extract_features("http://192.0.2.1/login")
        self.assertEqual(features["hostname_is_ip"], 1)
        self.assertEqual(features["uses_https"], 0)

    def test_https_in_path_does_not_change_scheme(self):
        features = extract_features(
            "http://example.com/https://another.example"
        )
        self.assertEqual(features["uses_https"], 0)


if __name__ == "__main__":
    unittest.main()
