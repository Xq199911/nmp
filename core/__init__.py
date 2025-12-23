# Core package for MP-KVM
from .clustering import OnlineManifoldCluster
from .integration import MPKVMManager, monkey_patch_attention_forward, patch_llama_attention
from .layers import ReconstructedAttention

"""
Core MP-KVM modules package.
"""


