import numpy as np
from alipy.query_strategy.LAL_RL.helpers import Minibatch
from alipy.query_strategy.LAL_RL import Agent
import torch
import random

def create_exp(n_state_estimation, size):
    state = np.random.rand(size, n_state_estimation)
    state.sort(axis=1)

    action = np.random.rand(size, 3)
    reward = np.random.rand(size)

    next_state = np.random.rand(size, n_state_estimation)
    next_state.sort(axis=1)

    next_actions = []
    for _ in range(size):
        tmp_action = np.random.rand(3, np.random.randint(1,15))
        next_actions.append(tmp_action)

    terminal = np.random.choice(np.array([True, False]), size, p=[0.2,0.8])
    indicies = np.arange(size)

    return Minibatch(state, action, reward, next_state, next_actions, terminal, indicies)


target_fc1_weight = torch.tensor([[-0.05675649,  0.17648868, -0.02854183, -0.03155166, -0.60082054,
         0.25689396,  0.33293173,  0.26671308,  0.4690515 ,  0.5224396 ],
       [-0.11017223,  0.13213323,  0.12984948, -0.5845376 , -0.135541  ,
         0.25626874,  0.05838481,  0.04353965, -0.25341144,  0.12187067],
       [-0.30838203,  0.4144176 ,  0.27508014,  0.1929062 , -0.30403644,
        -0.1440356 , -0.5757451 ,  0.24973704,  0.38726145,  0.49362615],
       [-0.44835222,  0.04433133, -0.20695591, -0.54300344, -0.5898859 ,
         0.17991348, -0.23132388,  0.2174614 , -0.6205266 ,  0.48317528],
       [-0.11216596,  0.40155753,  0.39487416,  0.10277759,  0.05142111,
         0.26602432,  0.20321551, -0.07677866,  0.32487893,  0.3200132]], dtype=torch.float)
target_fc1_weight = target_fc1_weight.t()
target_fc1_bias = torch.tensor([-9.999795e-06,  9.999895e-06, -9.999523e-06, -9.999949e-06,
        9.999858e-06,  9.999282e-06,  9.999460e-06, -9.999953e-06,
       -9.999941e-06, -9.999943e-06], dtype=torch.float)

target_fc2_weight = torch.tensor([[ -0.1764006 ,  0.31446508, -0.07420992,  0.44438004, -0.21578185],
       [-0.482137  ,  0.34307733, -0.24471383,  0.12973078, -0.1421859 ],
       [ 0.05975063, -0.2195936 ,  0.28606817,  0.17144513,  0.1842611 ],
       [-0.49450466, -0.49281135,  0.411566  , -0.01978724, -0.19733663],
       [-0.51208794,  0.08836523,  0.01090651, -0.06142015,  0.23303276],
       [ 0.47366178,  0.05331943,  0.3367835 ,  0.1933522 , -0.07338819],
       [ 0.49790257,  0.09264141,  0.43189374, -0.28663263, -0.34358728],
       [-0.41359997,  0.00746459, -0.19171803,  0.56222546, -0.3916049 ],
       [-0.39454144, -0.20304321,  0.18642984,  0.16282694,  0.38016695],
       [ 0.0396979 ,  0.26766342,  0.37911773, -0.05527054, -0.03452031],
       [-0.18065093,  0.16664892,  0.40414104,  0.00776202, -0.0280389 ],
       [ 0.10048284, -0.3935636 , -0.5688355 ,  0.17260148,  0.51975596],
       [-0.15965503,  0.54411453, -0.2353499 , -0.57537365, -0.31139547]],
      dtype=torch.float)
target_fc2_weight = target_fc2_weight.t()
target_fc2_bias = torch.tensor([-9.999980e-06,  9.999970e-06, -9.999945e-06, -9.999993e-06,
       -9.999973e-06], dtype=torch.float)

target_fc3_weight = torch.tensor([[0.17188057],
       [-0.30654308],
       [ 0.7373728 ],
       [ 0.24579573],
       [-0.3174925 ]], dtype=torch.float)
target_fc3_weight = target_fc3_weight.t()
target_fc3_bias = torch.tensor([-9.999998e-06], dtype=torch.float)

####################################################################

fc1_weight = torch.tensor([[-0.21255042, -0.39448613,  0.618815  ,  0.32926872, -0.32896513,
        -0.3390605 ,  0.28077   , -0.12843882, -0.2753594 , -0.07498472],
       [ 0.23484887,  0.14556831, -0.11372128, -0.2589378 , -0.35292098,
        -0.16477819, -0.21583056,  0.3444664 , -0.5230896 , -0.21501246],
       [-0.1448581 , -0.02317679, -0.39786083,  0.46703064,  0.2910301 ,
        -0.24814366, -0.45942405,  0.4982053 ,  0.30167276,  0.6208329 ],
       [-0.3104597 ,  0.3444692 ,  0.54587895,  0.19329756, -0.49225107,
        -0.18963152,  0.21761833, -0.26293415, -0.04597121,  0.38769388],
       [-0.47703108,  0.37163293,  0.08788786,  0.1126483 , -0.41250956,
         0.36662635, -0.3947447 ,  0.22105484, -0.11497063,  0.38037145]], dtype=torch.float)
fc1_weight = fc1_weight.t()
fc1_bias = torch.tensor([-0.00099998,  0.00099999, -0.00099995, -0.00099999,  0.00099999,
        0.00099993,  0.00099995, -0.001     , -0.00099999, -0.00099999], dtype=torch.float)

fc2_weight = torch.tensor([[ 0.46005732, -0.2674754 , -0.43970248, -0.07596845,  0.2663347 ],
       [ 0.40214205,  0.5255944 ,  0.14096802, -0.38779622,  0.30312234],
       [-0.23716222, -0.5437787 , -0.21813485,  0.12178235, -0.33744472],
       [ 0.5217554 ,  0.21558513,  0.3727172 ,  0.30464906,  0.2795167 ],
       [ 0.39570975,  0.56988996,  0.51195914, -0.27512664, -0.01874555],
       [ 0.5466669 ,  0.04350979, -0.11756837, -0.17134659, -0.07440609],
       [ 0.46749502, -0.00614011,  0.1508398 , -0.11931024, -0.4166432 ],
       [ 0.56491596, -0.0667619 , -0.53950375,  0.4282635 ,  0.223527  ],
       [-0.38147348, -0.3250605 , -0.37265405,  0.4271831 ,  0.5022133 ],
       [-0.08450157, -0.30369565, -0.00816751,  0.41797394,  0.42484635],
       [-0.18251145,  0.18261027,  0.35026866, -0.36323017,  0.14845777],
       [-0.5718769 , -0.07775211,  0.25246233, -0.4587126 ,  0.11813182],
       [-0.30683988,  0.42221242,  0.530479  , -0.4812235 , -0.38450134]],
      dtype=torch.float)
fc2_weight = fc2_weight.t()
fc2_bias = torch.tensor([-0.001     ,  0.001     , -0.00099999, -0.001     , -0.001], dtype=torch.float)

fc3_weight = torch.tensor([[ 0.4172718 ],
       [-0.1865383 ],
       [ 0.09723823],
       [ 0.8308312 ],
       [ 0.2608108]], dtype=torch.float)
fc3_weight = fc3_weight.t()
fc3_bias = torch.tensor([-0.001], dtype=torch.float)



agent = Agent(5, learning_rate=1e-3, batch_size=32, bias_average=0,
               target_copy_factor=0.01, gamma=1)

target_sd = agent.target_net.state_dict()
sd = agent.net.state_dict()

keys = list(target_sd.keys())

target_sd[keys[0]] = torch.nn.Parameter(target_fc1_weight)
sd[keys[0]] = torch.nn.Parameter(fc1_weight)
target_sd[keys[1]] = torch.nn.Parameter(target_fc1_bias)
sd[keys[1]] = torch.nn.Parameter(fc1_bias)
target_sd[keys[2]] = torch.nn.Parameter(target_fc2_weight)
sd[keys[2]] = torch.nn.Parameter(fc2_weight)
target_sd[keys[3]] = torch.nn.Parameter(target_fc2_bias)
sd[keys[3]] = torch.nn.Parameter(fc2_bias)
target_sd[keys[4]] = torch.nn.Parameter(target_fc3_weight)
sd[keys[4]] = torch.nn.Parameter(fc3_weight)
target_sd[keys[5]] = torch.nn.Parameter(target_fc3_bias)
sd[keys[5]] = torch.nn.Parameter(fc3_bias)

agent.net.load_state_dict(sd)
agent.target_net.load_state_dict(target_sd)

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
td_errors = agent.train(create_exp(5,32))

print("td_errors:", td_errors)