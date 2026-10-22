"""Regression cases for the false negatives found in the actual evaluation."""
import unittest

from reproduction.scoring import grade_math500


class MathScoringTests(unittest.TestCase):
    def test_bold_numeric_final_answer(self):
        self.assertTrue(grade_math500("Compute the result.", r"$\boxed{203}$", "The final answer is **203**."))

    def test_wrong_bold_numeric_answer(self):
        self.assertFalse(grade_math500("Compute the result.", r"$\boxed{203}$", "The final answer is **204**."))

    def test_degrees(self):
        self.assertTrue(grade_math500("Find the angle.", r"$\boxed{90^\circ}$", "The final answer is **90** degrees."))

    def test_algebraic_equivalence(self):
        self.assertTrue(grade_math500("Simplify.", r"$\boxed{\frac{1}{2}}$", r"Therefore, the final answer is $\boxed{0.5}$."))


if __name__ == "__main__":
    unittest.main()
