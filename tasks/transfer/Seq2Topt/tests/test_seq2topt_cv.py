import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from run_cv_enzsub import build_sequence_folds, sequence_groups  # noqa: E402

class SequenceFoldTests(unittest.TestCase):
    def test_duplicate_sequences_stay_in_one_fold(self):
        table = pd.DataFrame(
            {
                "sequence": ["ACDE", "ACDE", "FGHI", "JKLM", "JKLM", "NOPQ"],
                "topt": [20, 21, 30, 40, 41, 50],
            }
        )
        folds = build_sequence_folds(table, 3, 7, "sequence", 1022)
        groups = sequence_groups(table, "sequence", 1022)
        for group in set(groups.tolist()):
            self.assertEqual(len(set(folds[groups == group].tolist())), 1)
        self.assertEqual(sorted(set(folds.tolist())), [0, 1, 2])

if __name__ == "__main__":
    unittest.main()
