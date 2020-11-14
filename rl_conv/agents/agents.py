"""
@author:    Rilwan Adewoyin
@copyright: Copyright 2019, rlchat
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.02.20.
"""

import torch

from parlai.core.agents import (
    _load_opt_file, 
    get_agent_module, 
    add_task_flags_to_agent_opt,
    get_task_module)

from parlai.core.torch_generator_agent import (
    TorchGeneratorModel,
    TorchGeneratorAgent)

from parlai.core.utils import padded_tensor
from parlai.core.agents import _load_opt_file

from torch.autograd import Variable, backward
from torch.distributions import Categorical
from torch.nn import Module

from os.path import isfile


#define teacher (data provider )