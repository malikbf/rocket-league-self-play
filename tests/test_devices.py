"""Check backend policy independently of the host machine."""
import unittest
from unittest.mock import patch

from rocket_league_ai.training.devices import select_device


class DeviceTests(unittest.TestCase):
    def test_auto_policy_across_hardware(self):
        for cuda, mps in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(cuda=cuda, mps=mps), \
                 patch("torch.cuda.is_available", return_value=cuda), \
                 patch("torch.backends.mps.is_available", return_value=mps):
                self.assertEqual(select_device(), "cuda" if cuda else "cpu")
                expected = "cuda" if cuda else "mps" if mps else "cpu"
                self.assertEqual(select_device(for_ppo=False), expected)
                self.assertEqual(select_device("cpu"), "cpu")

    def test_ppo_rejects_unverified_mps(self):
        with self.assertRaisesRegex(ValueError, "unverified"):
            select_device("mps")

    def test_explicit_accelerator_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            for device in ("cuda", "mps"):
                with self.subTest(device=device), self.assertRaises(RuntimeError):
                    select_device(device, for_ppo=False)

    def test_explicit_accelerator_available(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_device("cuda"), "cuda")
            self.assertEqual(select_device("mps", for_ppo=False), "mps")

    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            select_device("typo")
