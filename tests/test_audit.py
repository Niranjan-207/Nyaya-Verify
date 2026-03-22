import unittest
import sys
import os

# Append src properly to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.audit.nli_judge import PairwiseAuditor

class TestNLIJudge(unittest.TestCase):
    def setUp(self):
        self.auditor = PairwiseAuditor()
        
    def tearDown(self):
        # Enforce VRAM cleanup identically as expected in production
        self.auditor.unload()
        
    def test_logic_flips_prioritization(self):
        chunk_2021 = {
            "text": "Platforms must remove illegal content within 36 hours.",
            "metadata": {"statute_year": 2021, "filename": "IT_Rules.pdf"}
        }
        chunk_2026 = {
            "text": "Platforms must remove illegal content within 3 hours.",
            "metadata": {"statute_year": 2026, "filename": "IT_Rules_Amendment.pdf"}
        }
        
        # Action test
        result = self.auditor.detect_logic_flips([chunk_2021, chunk_2026])
        
        self.assertTrue(result["ConflictFound"])
        self.assertGreater(result["ContradictionScore"], 0.8)
        self.assertEqual(result["PrioritizedChunk"]["metadata"]["statute_year"], 2026)
        
    def test_same_statute_ignoring(self):
        # Assert logic flips natively ignore identical reference bounds
        chunk_1 = {
            "text": "The content must be removed.",
            "metadata": {"statute_year": 2021, "filename": "IT_Rules.pdf"}
        }
        chunk_2 = {
            "text": "The content must not be removed.",
            "metadata": {"statute_year": 2021, "filename": "IT_Rules.pdf"}
        }
        
        result = self.auditor.detect_logic_flips([chunk_1, chunk_2])
        self.assertFalse(result["ConflictFound"])

if __name__ == '__main__':
    unittest.main()
