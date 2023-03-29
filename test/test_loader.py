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

    def test_gene_loader(self):
        gene = load_datasest(name="gene", kwargs={"sample_num": 500, "version": 5})
        self.assertEqual(gene["var_num"], 801)
        self.assertEqual(gene["sample_num"], 500)
        self.assertEqual(gene["name"], "Gene_s500_v5")
        with self.assertRaises(ValueError):
            load_datasest(name="gene", kwargs={"sample_num": 300, "version": 1})
        with self.assertRaises(ValueError):
            load_datasest(name="gene", kwargs={"sample_num": 500, "version": 32})

    def test_hailfinder_loader(self):
        hailfinder = load_datasest(
            name="hailfinder", kwargs={"index": 3, "sample_num": 500, "version": 5}
        )
        self.assertEqual(hailfinder["var_num"], 168)
        self.assertEqual(hailfinder["sample_num"], 500)
        self.assertEqual(hailfinder["name"], "HailFinder3_s500_v5")
        with self.assertRaises(ValueError):
            load_datasest(
                name="hailfinder", kwargs={"index": 18, "sample_num": 500, "version": 5}
            )
        with self.assertRaises(ValueError):
            load_datasest(
                name="hailfinder", kwargs={"index": 3, "sample_num": 300, "version": 5}
            )
        with self.assertRaises(ValueError):
            load_datasest(
                name="hailfinder", kwargs={"index": 3, "sample_num": 500, "version": 18}
            )

    def test_insurance_loader(self):
        ins = load_datasest(
            name="insurance", kwargs={"index": 1, "sample_num": 500, "version": 5}
        )
        self.assertEqual(ins["var_num"], 27)
        self.assertEqual(ins["sample_num"], 500)
        self.assertEqual(ins["name"], "Insurance_s500_v5")

    def test_jdk_loader(self):
        jdk = load_datasest(name="jdk", kwargs={})
        self.assertEqual(jdk["var_num"], 15)
        self.assertEqual(jdk["sample_num"], 37840)

    def test_link_loader(self):
        link = load_datasest(name="link", kwargs={"sample_num": 1000, "version": 1})
        self.assertEqual(link["var_num"], 724)
        self.assertEqual(link["sample_num"], 1000)
        self.assertEqual(link["name"], "link_s1000_v1")

    def test_mildew_loader(self):
        mildew = load_datasest(name="mildew", kwargs={"sample_num": 500, "version": 10})
        self.assertEqual(mildew["var_num"], 35)
        self.assertEqual(mildew["name"], "mildew_s500_v10")

    def test_munin1_loader(self):
        munin1 = load_datasest(name="munin1", kwargs={"sample_num": 500, "version": 10})
        self.assertEqual(munin1["var_num"], 189)
        self.assertEqual(munin1["name"], "munin1_s500_v10")

    def test_networking_loader(self):
        networking = load_datasest(name="networking", kwargs={})
        self.assertEqual(networking["var_num"], 17)
        self.assertEqual(networking["sample_num"], 415840)
        self.assertEqual(networking["name"], "networking")

    def test_pigs_loader(self):
        pigs = load_datasest(name="pigs", kwargs={"sample_num": 500, "version": 10})
        self.assertEqual(pigs["var_num"], 441)
        self.assertEqual(pigs["sample_num"], 500)
        self.assertEqual(pigs["name"], "pigs_s500_v10")

    def test_postgres_loader(self):
        postgres = load_datasest(name="postgres", kwargs={})
        self.assertEqual(postgres["var_num"], 21)
        self.assertEqual(postgres["sample_num"], 1506270)
        self.assertEqual(postgres["name"], "postgres")

    def test_realautompg_loader(self):
        real_auto_mpg = load_datasest(name="real_auto_mpg", kwargs={})
        self.assertEqual(real_auto_mpg["var_num"], 8)
        self.assertEqual(real_auto_mpg["sample_num"], 392)
        self.assertEqual(real_auto_mpg["name"], "real_auto_mpg")

    def test_realcites_loader(self):
        real_cites = load_datasest(name="real_cites", kwargs={})
        self.assertEqual(real_cites["var_num"], 7)
        self.assertEqual(real_cites["sample_num"], 7)
        self.assertEqual(real_cites["name"], "real_cites")

    def test_realyacht_loader(self):
        real_yacht = load_datasest(name="real_yacht", kwargs={})
        self.assertEqual(real_yacht["var_num"], 7)
        self.assertEqual(real_yacht["sample_num"], 308)
        self.assertEqual(real_yacht["name"], "real_yacht")

    def test_sachs_loader(self):
        sachs = load_datasest(name="sachs", kwargs={})
        self.assertEqual(sachs["var_num"], 11)
        self.assertEqual(sachs["sample_num"], 7466)
        self.assertEqual(sachs["name"], "sachs")

    def test_simulatedfeedback_loader(self):
        feedback = load_datasest(
            name="simulated_feedback", kwargs={"name": "Network5_amp", "version": 3}
        )
        self.assertEqual(feedback["var_num"], 5)
        self.assertEqual(feedback["sample_num"], 500)
        self.assertEqual(feedback["name"], "sim-03.Network5_amp.continuous")


if __name__ == "__main__":
    unittest.main()
