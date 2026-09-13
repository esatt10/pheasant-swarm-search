"""Pheasant Swarm Search.

A planner divides a topic into questions, research workers search literature
in parallel, and a coverage auditor identifies gaps. Source material is saved
in pheasant-kb for the next phase of a project. Optional benchmark comparisons
measure how well fresh agents retrieve and answer from that knowledge base.

The five questions this package keeps apart, and never blends into one score:
collection, persistence, retrieval, answering, and learning.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
