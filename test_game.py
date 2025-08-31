import unittest
import numpy as np
from unittest.mock import patch
from src.game import Game2048

class TestGame2048(unittest.TestCase):

    def setUp(self):
        self.game = Game2048()

    def test_initial_board(self):
        # Ensure the board starts with exactly two tiles
        non_zero_count = np.count_nonzero(self.game.board)
        self.assertEqual(non_zero_count, 2)

    def test_add_random_tile(self):
        # Add a random tile and check if the count increases
        initial_count = np.count_nonzero(self.game.board)
        self.game.add_random_tile()
        new_count = np.count_nonzero(self.game.board)
        self.assertEqual(new_count, initial_count + 1)

    def test_slide_and_merge(self):
        # Test sliding and merging logic
        row = np.array([2, 2, 0, 0])
        expected = np.array([4, 0, 0, 0])
        result = self.game.slide_and_merge(row)
        np.testing.assert_array_equal(result, expected)

        row = np.array([2, 2, 2, 2])
        expected = np.array([4, 4, 0, 0])
        result = self.game.slide_and_merge(row)
        np.testing.assert_array_equal(result, expected)

    @patch.object(Game2048, 'add_random_tile')
    def test_move_left(self, mock_add_random_tile):
        # Test moving left
        self.game.board = np.array([
            [2, 2, 0, 0],
            [4, 0, 4, 0],
            [0, 0, 0, 0],
            [2, 2, 2, 2]
        ])
        self.game.move('left')
        expected = np.array([
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 4, 0, 0]
        ])
        np.testing.assert_array_equal(self.game.board, expected)

    def test_game_over(self):
        # Test game over condition
        self.game.board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        self.assertTrue(self.game.is_game_over())

    def test_not_game_over(self):
        # Test game not over condition
        self.game.board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 0]
        ])
        self.assertFalse(self.game.is_game_over())

if __name__ == "__main__":
    unittest.main()