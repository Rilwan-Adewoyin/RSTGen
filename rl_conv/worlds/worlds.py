import torch

from parlai.core.worlds import (
    MultiAgentDialogWorld)

Action = namedtuple('Action', ['id', 'action', 'responses'])

"""
dict, containing the agent's response for the action.
"""


class RLMConvWorld(MultiAgentDialogWorld):
    
    def __init__.(self, opt:Opt, agents, teacher, shared=None  ):
        
        """"""
        self.id = 'RLMconvWorld'
        self.episode_batch = None
        self.active_agent = active_agent
        #self.static_agent = static_agent
        agents = teacher + [ active_agent]

        super(RLMConvWorld, self).__init__(opt, agents, shared)    


    def parley():
