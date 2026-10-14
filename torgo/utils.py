"""
==============================================================================
utils.py — Shared Utilities for TORGO ML Pipeline
==============================================================================
Checklist tracker and helper functions used across all phases.
"""


class PipelineChecklist:
    """Tracks completion status of all pipeline steps."""

    def __init__(self):
        self.steps = {}

    def mark_done(self, step_id, description):
        """Mark a pipeline step as completed."""
        self.steps[step_id] = {"description": description, "status": "DONE"}
        print(f"  [DONE] [{step_id}] {description}")

    def print_summary(self):
        """Print the full pipeline checklist."""
        print("\n" + "=" * 70)
        print("  PIPELINE CHECKLIST — FINAL STATUS")
        print("=" * 70)
        for step_id, info in self.steps.items():
            status_icon = "[DONE]" if info["status"] == "DONE" else "[    ]"
            print(f"  {status_icon}  [{step_id}]  {info['description']}")
        print("=" * 70)


def print_phase_header(phase_num, title):
    """Print a formatted phase header."""
    print("\n" + "=" * 70)
    print(f"  PHASE {phase_num}: {title}")
    print("=" * 70)
