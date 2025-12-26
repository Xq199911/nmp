# Core package for MP-KVM
from .clustering import OnlineManifoldCluster
# use clean integration implementation to avoid shadowed/duplicate definitions
from .integration_clean import MPKVMManager, monkey_patch_attention_forward, patch_llama_attention
from .layers import ReconstructedAttention

"""
Core MP-KVM modules package.
"""


