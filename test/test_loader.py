import unittest
from causalbench.utils.helper import load_datasest


class TestLoaderMethods(unittest.TestCase):
    def test_alarm_loader(self):
        alarm = load_datasest(
            name="alarm", kwargs={"index": 10, "sample_num": 5000, "version": 5}
        )
        self.assertEqual(alarm["name"], "Alarm10_s5000_v5")
        self.assertEqual(alarm["var_num"], 370)

    def test_barley_loader(self):
        barley = load_datasest(name="barley", kwargs={"sample_num": 500, "version": 10})
        self.assertEqual(barley["var_num"], 48)
        self.assertEqual(barley["name"], "Barley_s500_v10")

    def test_child_loader(self):
        child = load_datasest(
            name="child", kwargs={"index": 5, "sample_num": 1000, "version": 5}
        )
        self.assertEqual(child["var_num"], 100)
        self.assertEqual(child["sample_num"], 1000)
        self.assertEqual(child["name"], "Child5_s1000_v5")

    def test_dataverse_loader(self):
        dataverse = load_datasest(
            name="dataverse",
            kwargs={
                "with_hidden_var": True,
                "is_big": False,
                "max_parent_num": 5,
                "version": 5,
            },
        )
        self.assertEqual(dataverse["var_num"], 18)
        self.assertEqual(dataverse["sample_num"], 500)
        self.assertEqual(dataverse["name"], "G5_v5_confounders_numdata.csv")

        dataverse_big = load_datasest(
            name="dataverse",
            kwargs={
                "with_hidden_var": False,
                "is_big": True,
                "max_parent_num": 3,
                "version": 1,
            },
        )
        self.assertEqual(dataverse_big["var_num"], 101)
        self.assertEqual(dataverse_big["sample_num"], 500)
        self.assertEqual(dataverse_big["name"], "Big_G3_v1_numdata.csv")

    def test_dream4_loader(self):
        dream4 = load_datasest(name="dream4", kwargs={"version": 4})
        self.assertEqual(dream4["var_num"], 100)
        self.assertEqual(dream4["sample_num"], 100)
        self.assertEqual(dream4["name"], "dream4_4")
        with self.assertRaises(ValueError):
            load_datasest(name="dream4", kwargs={"version": 8})

    # def test_isupper(self):
    #     self.assertTrue("FOO".isupper())
    #     self.assertFalse("Foo".isupper())

    # def test_split(self):
    #     s = "hello world"
    #     self.assertEqual(s.split(), ["hello", "world"])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == "__main__":
    unittest.main()
