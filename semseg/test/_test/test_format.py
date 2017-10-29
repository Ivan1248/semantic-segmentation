import unittest
import processing
from test import test_support


class MyTestCase1(unittest.TestCase):
    # Only use setUp() and tearDown() if necessary

    def setUp(self):
        # ... code to execute in preparation for tests ...
        pass

    def tearDown(self):
        # ... code to execute to clean up after tests ...
        pass

    def test_feature_one(self):
        # Test feature one.
        pass

    def test_feature_two(self):
        # Test feature two.
        pass


class MyTestCase2(unittest.TestCase):
    # ... same structure as MyTestCase1 ...
    pass


def test_main():
    test_support.run_unittest(MyTestCase1, MyTestCase2)


if __name__ == '__main__':
    test_main()
