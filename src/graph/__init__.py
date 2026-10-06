"""AgentCore Platform v1.0"""

# Registry entry-point export. config/agent.yaml declares the dotted entry
# point src.graph.graph.InsuranceClaimsBottleneckDiscoveryAgent; re-exporting
# the class from the package root keeps `import src.graph` sufficient for any
# caller that resolves the manifest by module rather than by dotted path.
from .graph import InsuranceClaimsBottleneckDiscoveryAgent

__all__ = ["InsuranceClaimsBottleneckDiscoveryAgent"]
