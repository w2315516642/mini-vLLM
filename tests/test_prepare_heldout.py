import unittest
from benchmarks.prepare_heldout import select, family


def row(i, text, role="human"):
    return {"id": i, "conversations": [{"from": role, "value": text}]}


class SelectionTest(unittest.TestCase):
    def test_exclusion_and_dedup(self):
        excluded = [row("train_0", "abcd training")]
        source = [row("train_9", "unseen sibling"),
                  row("copy_0", "abcd training"),
                  row("prefix_0", "abcd different"),
                  row("bad_0", "valid-looking", "gpt"),
                  row("short_0", "x"), row("a_0", "good one"),
                  row("b_0", "good two"), row("c_0", "else")]
        result = select(source, excluded, list, 2, 4, 42)
        self.assertEqual({r["prompt"][:4] for r in result}, {"good", "else"})
        self.assertEqual(result, select(source, excluded, list, 2, 4, 42))

    def test_insufficient_fails(self):
        with self.assertRaises(ValueError):
            select([row("a_0", "abc")], [], list, 2, 3, 42)

    def test_family(self):
        self.assertEqual(family("tgKByb7_9"), "tgKByb7")
        self.assertEqual(family("unsplit"), "unsplit")


if __name__ == "__main__":
    unittest.main()
