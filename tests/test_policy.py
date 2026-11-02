"""Behavioral tests for confirmation, dismissal, and rearming."""

import math
import unittest

from privacy_guard.policy import AlertPolicy, HoldTrigger, ArmState


class AlertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AlertPolicy()

    def open_alert(self, start=0.0):
        self.assertFalse(self.policy.update(2, start))
        self.assertTrue(self.policy.update(2, start + 0.8))
        self.assertTrue(self.policy.alert_visible)

    def clear_scene(self, start=1.0):
        for offset in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            self.assertFalse(self.policy.update(1, start + offset))

    def test_confirms_at_boundary_and_opens_only_once(self):
        self.assertFalse(self.policy.update(2, 0.0))
        self.assertFalse(self.policy.update(2, 0.799))
        self.assertTrue(self.policy.update(2, 0.8))
        self.assertFalse(self.policy.update(2, 0.9))
        self.assertTrue(self.policy.incident_latched)

    def test_one_person_and_empty_scene_do_not_alert(self):
        for index in range(30):
            self.assertFalse(self.policy.update(index % 2, index / 10))
        self.assertFalse(self.policy.alert_visible)

    def test_more_than_two_faces_also_alerts(self):
        self.assertFalse(self.policy.update(3, 0.0))
        self.assertTrue(self.policy.update(5, 0.8))

    def test_brief_false_positive_restarts_confirmation(self):
        self.assertFalse(self.policy.update(2, 0.0))
        self.assertFalse(self.policy.update(1, 0.6))
        self.assertFalse(self.policy.update(2, 0.7))
        self.assertFalse(self.policy.update(2, 1.4))
        self.assertTrue(self.policy.update(2, 1.6))

    def test_dismiss_suppresses_continued_multiple_faces(self):
        self.open_alert()
        self.policy.dismiss()
        self.assertFalse(self.policy.alert_visible)
        for index in range(1, 40):
            self.assertFalse(self.policy.update(2, 0.8 + index / 2))

    def test_brief_dip_does_not_rearm(self):
        self.open_alert()
        self.policy.dismiss()
        for timestamp in (1.0, 2.0, 3.0, 3.9):
            self.assertFalse(self.policy.update(1, timestamp))
        self.assertFalse(self.policy.update(2, 4.0))
        self.assertFalse(self.policy.update(2, 4.8))
        self.assertTrue(self.policy.incident_latched)

    def test_sustained_clear_rearms_at_exact_boundary(self):
        self.open_alert()
        self.policy.dismiss()
        self.clear_scene()
        self.assertFalse(self.policy.incident_latched)
        self.assertFalse(self.policy.update(2, 4.1))
        self.assertTrue(self.policy.update(2, 5.0))

    def test_returning_face_restarts_clear_window(self):
        self.open_alert()
        self.policy.dismiss()
        for timestamp in (1.0, 2.0, 3.0):
            self.policy.update(1, timestamp)
        self.policy.update(2, 3.5)
        for timestamp in (4.0, 5.0, 6.0):
            self.policy.update(1, timestamp)
        self.assertTrue(self.policy.incident_latched)
        self.policy.update(1, 7.0)
        self.assertFalse(self.policy.incident_latched)

    def test_clear_scene_does_not_automatically_close_window(self):
        self.open_alert()
        self.clear_scene()
        self.assertTrue(self.policy.alert_visible)
        self.assertFalse(self.policy.incident_latched)
        self.policy.dismiss()
        self.assertFalse(self.policy.alert_visible)
        self.assertFalse(self.policy.incident_latched)
        self.assertFalse(self.policy.update(2, 4.1))
        self.assertTrue(self.policy.update(2, 5.0))

    def test_old_visible_window_does_not_duplicate_on_new_incident(self):
        self.open_alert()
        self.clear_scene()
        self.assertFalse(self.policy.update(2, 4.1))
        self.assertFalse(self.policy.update(2, 5.0))
        self.assertTrue(self.policy.alert_visible)
        self.policy.dismiss()
        self.assertTrue(self.policy.incident_latched)
        self.assertFalse(self.policy.update(2, 5.5))
        self.assertFalse(self.policy.update(2, 6.0))

    def test_dismiss_during_partial_clear_keeps_incident_latched(self):
        self.open_alert()
        self.policy.update(1, 1.0)
        self.policy.dismiss()
        self.assertTrue(self.policy.incident_latched)
        self.assertFalse(self.policy.update(2, 1.5))
        self.assertFalse(self.policy.update(2, 2.3))

    def test_no_method_opens_alert_without_a_fresh_frame(self):
        self.policy.update(2, 0.0)
        self.assertFalse(self.policy.alert_visible)
        # A later fresh frame after a camera gap starts a new window.
        self.assertFalse(self.policy.update(2, 30.0))
        self.assertTrue(self.policy.update(2, 30.8))

    def test_missing_frames_do_not_count_toward_clearance(self):
        self.open_alert()
        self.policy.dismiss()
        self.policy.update(1, 1.0)
        self.policy.update(1, 30.0)
        self.assertTrue(self.policy.incident_latched)
        self.assertFalse(self.policy.update(2, 30.5))

    def test_irregular_fresh_frame_intervals_can_confirm(self):
        for timestamp in (10.0, 10.03, 10.2, 10.61, 10.79):
            self.assertFalse(self.policy.update(2, timestamp))
        self.assertTrue(self.policy.update(2, 10.81))

    def test_repeated_timestamp_adds_no_confirmation_time(self):
        for _ in range(10):
            self.assertFalse(self.policy.update(2, 1.0))
        self.assertFalse(self.policy.update(2, 1.7))
        self.assertTrue(self.policy.update(2, 1.81))

    def test_backward_time_is_rejected_without_corrupting_state(self):
        self.policy.update(2, 5.0)
        with self.assertRaises(ValueError):
            self.policy.update(0, 4.0)
        self.assertTrue(self.policy.update(2, 5.9))

    def test_invalid_counts_are_rejected_without_modifying_state(self):
        self.policy.update(2, 0.0)
        for invalid_count in (-1, 2.0, True, None, "2"):
            with self.subTest(count=invalid_count):
                with self.assertRaises(ValueError):
                    self.policy.update(invalid_count, 0.4)
        self.assertTrue(self.policy.update(2, 0.8))

    def test_invalid_timestamps_are_rejected(self):
        for invalid_time in (math.nan, math.inf, -math.inf, None, True, "0"):
            with self.subTest(now=invalid_time):
                with self.assertRaises(ValueError):
                    self.policy.update(2, invalid_time)
        self.assertFalse(self.policy.alert_visible)

    def test_reset_clears_visible_and_latched_state_and_clock(self):
        self.open_alert(10.0)
        self.policy.reset()
        self.assertFalse(self.policy.alert_visible)
        self.assertFalse(self.policy.incident_latched)
        self.assertFalse(self.policy.update(2, 0.0))
        self.assertTrue(self.policy.update(2, 0.8))

    def test_reset_discards_unconfirmed_candidate(self):
        self.policy.update(2, 0.0)
        self.policy.reset()
        self.assertFalse(self.policy.update(2, 0.8))
        self.assertFalse(self.policy.update(2, 1.0))
        self.assertTrue(self.policy.update(2, 1.7))

    def test_zero_duration_settings_are_supported(self):
        policy = AlertPolicy(confirm_seconds=0, clear_seconds=0)
        self.assertTrue(policy.update(2, 0.0))
        policy.dismiss()
        self.assertFalse(policy.update(1, 0.1))
        self.assertTrue(policy.update(2, 0.2))

    def test_custom_threshold(self):
        policy = AlertPolicy(face_threshold=3)
        self.assertFalse(policy.update(2, 0.0))
        self.assertFalse(policy.update(3, 0.1))
        self.assertTrue(policy.update(3, 0.9))

    def test_invalid_configuration(self):
        cases = (
            {"confirm_seconds": -1},
            {"confirm_seconds": math.nan},
            {"clear_seconds": -1},
            {"clear_seconds": math.inf},
            {"max_frame_gap_seconds": 0},
            {"max_frame_gap_seconds": -1},
            {"face_threshold": 1},
            {"face_threshold": True},
            {"face_threshold": 2.0},
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    AlertPolicy(**config)


class HoldTriggerTests(unittest.TestCase):
    def test_turns_on_only_after_sustained_true(self):
        # Frames are spaced within max_frame_gap_seconds (fresh-frame cadence).
        trigger = HoldTrigger(on_seconds=2.0, off_seconds=3.0)
        self.assertFalse(trigger.update(True, 0.0))
        self.assertFalse(trigger.update(True, 1.0))
        self.assertFalse(trigger.update(True, 1.9))
        self.assertTrue(trigger.update(True, 2.0))
        self.assertTrue(trigger.active)

    def test_turns_off_only_after_sustained_false(self):
        trigger = HoldTrigger(on_seconds=1.0, off_seconds=3.0)
        trigger.update(True, 0.0)
        self.assertTrue(trigger.update(True, 1.0))
        self.assertTrue(trigger.update(False, 2.0))
        self.assertTrue(trigger.update(False, 3.0))
        self.assertTrue(trigger.update(False, 4.0))
        self.assertFalse(trigger.update(False, 5.0))

    def test_brief_dropout_does_not_turn_off(self):
        trigger = HoldTrigger(on_seconds=0.5, off_seconds=3.0)
        trigger.update(True, 0.0)
        trigger.update(True, 0.5)
        self.assertTrue(trigger.active)
        trigger.update(False, 1.0)
        trigger.update(True, 1.5)  # returned before off window elapsed
        self.assertTrue(trigger.update(True, 2.0))

    def test_frame_gap_restarts_windows_without_flipping_state(self):
        trigger = HoldTrigger(on_seconds=1.0, off_seconds=2.0, max_frame_gap_seconds=1.0)
        self.assertFalse(trigger.update(True, 0.0))
        self.assertFalse(trigger.update(True, 5.0))  # gap resets the on window
        self.assertFalse(trigger.active)
        self.assertTrue(trigger.update(True, 6.0))

    def test_rejects_backward_time(self):
        trigger = HoldTrigger(on_seconds=1.0, off_seconds=1.0)
        trigger.update(True, 5.0)
        with self.assertRaises(ValueError):
            trigger.update(True, 4.0)


class ArmStateTests(unittest.TestCase):
    def test_starts_armed(self):
        self.assertTrue(ArmState().armed)

    def test_escape_disarm_needs_manual_rearm(self):
        state = ArmState()
        state.disarm()
        self.assertFalse(state.armed)
        self.assertFalse(state.tick(10_000.0))  # never re-arms on its own
        state.arm()
        self.assertTrue(state.armed)

    def test_snooze_rearms_at_deadline(self):
        state = ArmState()
        state.snooze(now=100.0, seconds=60.0)
        self.assertFalse(state.armed)
        self.assertAlmostEqual(state.remaining(130.0), 30.0)
        self.assertFalse(state.tick(159.9))
        self.assertTrue(state.tick(160.0))
        self.assertTrue(state.armed)

    def test_remaining_is_zero_when_not_snoozed(self):
        state = ArmState()
        self.assertEqual(state.remaining(0.0), 0.0)
        state.disarm()
        self.assertEqual(state.remaining(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
