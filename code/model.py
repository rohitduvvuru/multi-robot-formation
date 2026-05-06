import torch
from torch import nn
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_cluster import radius_graph

import math


class ModGNNConv(MessagePassing):
    propagate_type = {"x": Tensor}

    def __init__(self, nn, aggr="mean", **kwargs):
        super(ModGNNConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.nn)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(x_j - x_i)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GNNBranch(nn.Module):
    def __init__(self, in_features, msg_features, out_features, activation):
        nn.Module.__init__(self)

        self.nns = self.generate_nn_instance(
            in_features, msg_features, out_features, activation
        )
        self.gnn = ModGNNConv(self.nns["gnn"], aggr="add").jittable()

    def generate_nn_instance(self, in_features, msg_features, out_features, activation):
        return torch.nn.ModuleDict(
            {
                "encoder": torch.nn.Sequential(
                    torch.nn.Linear(in_features, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, msg_features),
                ),
                "gnn": torch.nn.Sequential(
                    torch.nn.Linear(msg_features, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                ),
                "post": torch.nn.Sequential(
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, out_features),
                ),
            }
        )

    def forward(self, p: torch.Tensor, x: torch.Tensor, comm_radius: torch.Tensor):
        assert x.ndim == 3  # batch and features
        assert p.ndim == 3  # batch and positions
        batch_size = x.shape[0]
        n_agents = x.shape[1]

        encoding_out = self.nns["encoder"](x)

        b = torch.arange(0, batch_size, dtype=torch.int64, device=x.device)
        batch = torch.repeat_interleave(b, n_agents)
        edge_index = radius_graph(
            p.reshape(-1, p.shape[-1]), batch=batch, r=comm_radius[0], loop=True
        )
        gnn_in = encoding_out.reshape(-1, encoding_out.shape[-1])
        gnn_out = self.gnn(gnn_in, edge_index).view(batch_size, n_agents, -1)

        return self.nns["post"](gnn_out)


class Model(nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, nameW, **cfg):
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space

        self.n_agents = obs_space["pos"].shape[0]
        self.outputs_per_agent = int(num_outputs / self.n_agents)

        activation = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }[cfg["activation"]]

        self.comm_range = torch.Tensor([cfg["comm_range"]])

        self.gnn = GNNBranch(6, cfg["msg_features"], self.outputs_per_agent, activation)
        self.gnn_value = GNNBranch(6, cfg["msg_features"], 1, activation)
        self.use_beta = True

    def forward(self, input_dict, state, seq_lens):
        pos = input_dict["pos"]
        vel = input_dict["vel"]
        goal = input_dict["goal"]
        x = torch.cat([goal - pos, pos, pos + vel], dim=-1)
        outputs = self.gnn(pos, x, self.comm_range)
        values = self.gnn_value(pos, x, self.comm_range)
        self._cur_value = values.view(-1, self.n_agents)

        return outputs.view(-1, self.n_agents * self.outputs_per_agent), state

    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


class Agent(nn.Module):
    def __init__(self, env, config):
        super(Agent, self).__init__()

        self.obs_space = env.observation_space
        self.action_space = env.action_space

        model_config = config['model']['custom_model_config']
        # num_sub_agent = config['env_config']['n_subteam_agents']
        name = 'model'
        num_outputs = len(env.action_space)*env.action_space[0].shape[0]*2
        model = Model(env.observation_space, env.action_space, num_outputs, model_config, name,\
                      **model_config)

        self.model = model


    def set_comm_range(self, comm_range: float):
        self.model.comm_range = torch.tensor([comm_range], device=self.model.comm_range.device)

    def format_input(self, x, device):
        # format from dict to tensor input
        obs = x

        key_list = ['pos', 'vel', 'goal']
        concat_list = {}
        for key in key_list:
            concat_list[key] = []
            for obs_instance in obs:
                concat_list[key].append(obs_instance[key])


        input_dict = {}
        for key in key_list:
            input_dict[key] = torch.tensor(concat_list[key]).to(device)
        return input_dict

    def get_value(self, x):
        self.model(x, None, None)
        return self.model.value_function()

    def get_action_and_value(self, x, action=None):
        logits, state = self.model(x, None, None)
        bs = logits.shape[0]                            # remark: batch size = num_envs

        # make logits positive only (restriction for alpha and beta)
        x = torch.clamp(logits, math.log(1e-6), -math.log(1e-6))
        x = torch.log(torch.exp(x) + 1.0) + 1.0
        
        x = torch.reshape(x, (bs, -1, self.action_space[0].shape[0]*2))     # separate output per agent (dim=1)
        x = torch.permute(x, (1, 0, 2))                                     # order by agent then batch
        alpha, beta = torch.chunk(x, 2, dim=-1)                             # split x into alpha and beta

        # for each agent: generate action distribution, sample action, and compute logp and entropy
        actions = []
        logps = []
        entropys = []
        for idx, (agent_alpha, agent_beta, agent_action_space) in enumerate(zip(alpha, beta, self.action_space)):
            agent_probs = torch.distributions.Beta(concentration1=agent_alpha, concentration0=agent_beta)
            agent_low = torch.tensor(agent_action_space.low).to(x.device)
            agent_high = torch.tensor(agent_action_space.high).to(x.device)

            if action is None:
                agent_action = agent_probs.rsample().to(x.device)                   # reparameterization trick: has gradient
            else:
                agent_action = (action[:,idx,:] - agent_low) / (agent_high - agent_low)
            agent_logp = torch.sum(agent_probs.log_prob(agent_action), dim=-1)
            agent_entropy = torch.sum(agent_probs.entropy(), dim=-1)            # remark: differential entropy can be negative

            agent_action = agent_action * (agent_high - agent_low) + agent_low  # squash from [0,1] to [low,high]

            actions.append(agent_action)
            logps.append(agent_logp)
            entropys.append(agent_entropy)

        actions = torch.stack(actions, dim=1)       # shape: (bs, n_agents, 2)
        logps = torch.stack(logps, dim=1)           # shape: (bs, n_agents)
        entropys = torch.stack(entropys, dim=1)     # shape: (bs, n_agents)
        values = self.model.value_function()        # shape: (bs, n_agents)
        return actions, logps, entropys, values
