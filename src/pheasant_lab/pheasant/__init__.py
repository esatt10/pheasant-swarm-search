"""The Pheasant MCP client, and the adapters that give the lab a stable shape.

The lab's internal request and response shapes do not change when the server's
tool names do. That is deliberate: a lab that hard-codes tool names discovers
the mismatch during a paid run, and Pheasant's own docs say to read the
readiness contract before assuming a capability.
"""

from .capabilities import CapabilityMap, CapabilityResolution, MissingCapability
from .client import PheasantClient, PheasantSession
from .protocol import JsonRpcError, McpToolError, ProtocolError
from .receipts import IngestReceipt, ReceiptLedger
from .retrieval import SearchRequest, SearchResponse, SearchResult

__all__ = [
    "CapabilityMap",
    "CapabilityResolution",
    "IngestReceipt",
    "JsonRpcError",
    "McpToolError",
    "MissingCapability",
    "PheasantClient",
    "PheasantSession",
    "ProtocolError",
    "ReceiptLedger",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
