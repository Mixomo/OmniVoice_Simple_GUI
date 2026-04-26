import logging
import sys
import torch
import collections
import importlib.util

logger = logging.getLogger(__name__)

def apply_triton_windows_patch():
    """Robustly force Triton availability on Windows for PyTorch 2.8.0+ Inductor."""
    if sys.platform != "win32":
        return

    # 1. Ensure triton is available
    if importlib.util.find_spec("triton") is None:
        logger.warning("Triton package not found. Triton fixes will not be applied.")
        return

    try:
        import triton
        import triton.compiler.compiler as tc
    except ImportError:
        return

    # 2. Patch AttrsDescriptorWrapper to return plain dicts
    try:
        import torch._inductor.runtime.hints as inductor_hints
        def patched_attrs_wrapper(divisible_by_16=None, equal_to_1=None):
            return {
                "divisible_by_16": tuple(divisible_by_16) if divisible_by_16 else (),
                "equal_to_1": tuple(equal_to_1) if equal_to_1 else ()
            }
        inductor_hints.AttrsDescriptorWrapper = patched_attrs_wrapper
    except ImportError:
        pass

    # 2.5 Patch Triton CompiledKernel/Metadata to avoid 'cluster_dims' and 'launch_hook' errors
    try:
        import torch._inductor.runtime.triton_heuristics as th
        
        # Safety for missing hooks in some Torch/Triton versions
        if hasattr(th, "CompiledKernel"):
            if not hasattr(th.CompiledKernel, "launch_enter_hook"):
                th.CompiledKernel.launch_enter_hook = lambda *args, **kwargs: None
            if not hasattr(th.CompiledKernel, "launch_exit_hook"):
                th.CompiledKernel.launch_exit_hook = lambda *args, **kwargs: None

        orig_make_launcher = th.TritonCompileResult.make_launcher
        
        def patched_make_launcher(self):
            binary = self.kernel
            # metadata fix for older/newer triton mismatches
            if hasattr(binary, "metadata") and not hasattr(binary.metadata, "cluster_dims"):
                try:
                    # Try setting it directly
                    setattr(binary.metadata, "cluster_dims", (1, 1, 1))
                except (AttributeError, TypeError):
                    # Handle immutable namedtuples by recreating them
                    try:
                        meta_dict = binary.metadata._asdict()
                        if "cluster_dims" not in meta_dict:
                            meta_dict["cluster_dims"] = (1, 1, 1)
                        if "num_ctas" not in meta_dict:
                            meta_dict["num_ctas"] = 1
                        NewMeta = collections.namedtuple("KernelMetadata", sorted(meta_dict.keys()))
                        binary.metadata = NewMeta(**meta_dict)
                    except Exception:
                        pass
            return orig_make_launcher(self)
            
        th.TritonCompileResult.make_launcher = patched_make_launcher
    except Exception as e:
        logger.debug(f"Triton heuristic patch failed: {e}")
        pass

    # 3. Patch triton_key to prevent metadata errors
    if not hasattr(tc, "triton_key"):
        tc.triton_key = lambda *args, **kwargs: "windows_triton_key"
    
    t_comp = sys.modules.get("triton.compiler")
    if t_comp and not hasattr(t_comp, "triton_key"):
        t_comp.triton_key = tc.triton_key

    # 4. Force Torch detection flags
    import torch.utils._triton
    torch.utils._triton.has_triton = lambda: True
    torch.utils._triton.has_triton_package = lambda: True
    
    try:
        import torch._inductor.utils as inductor_utils
        inductor_utils.has_triton = lambda *args, **kwargs: True
        if hasattr(inductor_utils, "is_triton_available"):
            inductor_utils.is_triton_available = lambda *args, **kwargs: True
    except ImportError:
        pass
    
    # Enable Triton in Inductor config (if the flag exists)
    import torch._inductor.config as inductor_config
    if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "enabled"):
        inductor_config.triton.enabled = True

    # 5. Fix Dynamo NameError for 'triton' in record_compilation_metrics
    try:
        import torch._dynamo.utils as dynamo_utils
        if "triton" not in dynamo_utils.__dict__:
            dynamo_utils.triton = triton
    except (ImportError, AttributeError):
        pass

    logger.info("Applied comprehensive Triton/Inductor patches for Windows compatibility.")

def apply_flex_attention_patch():
    """
    Workaround for sm_86/sm_89 99 KB shared-memory limit in flex_attention.
    Only applies to RTX 30/40 series cards that have this limitation.
    """
    if not torch.cuda.is_available():
        return

    # Check compute capability
    major, minor = torch.cuda.get_device_capability()
    # 8.6 = Ampere (RTX 30), 8.9 = Ada Lovelace (RTX 40)
    # These cards have a strict 99KB limit per thread block.
    is_affected_gpu = (major == 8 and (minor == 6 or minor == 9))
    
    if not is_affected_gpu:
        logger.info(f"GPU compute capability {major}.{minor} detected. Skipping flex_attention 32x32 patch.")
        return

    try:
        import transformers.integrations.flex_attention as _fa
        _original = _fa.compile_friendly_flex_attention
        _kernel_opts = {
            "BLOCK_M": 32, "BLOCK_N": 32,
            "BLOCK_M1": 32, "BLOCK_N1": 32,
            "BLOCK_M2": 32, "BLOCK_N2": 32,
            "num_stages": 2, "num_warps": 4,
        }

        def _patched(query, key, value, training=False, **kwargs):
            opts = dict(kwargs.get("kernel_options") or {})
            for k, v in _kernel_opts.items():
                opts.setdefault(k, v)
            kwargs["kernel_options"] = opts
            return _original(query, key, value, training=training, **kwargs)

        _fa.compile_friendly_flex_attention = _patched
        logger.info(f"GPU SM {major}.{minor} detected: flex_attention 32x32 patch applied (VRAM safety).")
    except (ImportError, AttributeError):
        pass
