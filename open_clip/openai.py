import os
import warnings
from typing import List, Optional, Union
import torch
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url
__all__ = ['list_openai_models', 'load_openai_model']

def list_openai_models() -> List[str]:
    return list_pretrained_models_by_tag('openai')

def load_openai_model(name: str, precision: Optional[str]=None, device: Optional[Union[str, torch.device]]=None, jit: bool=True, cache_dir: Optional[str]=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'
    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f'Model {name} not found; available models = {list_openai_models()}')
    try:
        model = torch.jit.load(model_path, map_location=device if jit else 'cpu').eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f'File {model_path} is not a JIT archive. Loading as a state dict instead')
            jit = False
        state_dict = torch.load(model_path, map_location='cpu')
    if not jit:
        cast_dtype = get_cast_dtype(precision)
        try:
            model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
        except KeyError:
            sd = {k[7:]: v for (k, v) in state_dict['state_dict'].items()}
            model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)
        model = model.to(device)
        if precision.startswith('amp') or precision == 'fp32':
            model.float()
        elif precision == 'bf16':
            convert_weights_to_lp(model, dtype=torch.bfloat16)
        return model
    device_holder = torch.jit.trace(lambda : torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes('prim::Constant') if 'Device' in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, 'graph') else []
        except RuntimeError:
            graphs = []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    if precision == 'fp32':
        float_holder = torch.jit.trace(lambda : torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, 'graph') else []
            except RuntimeError:
                graphs = []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    model.visual.image_size = model.input_resolution.item()
    return model