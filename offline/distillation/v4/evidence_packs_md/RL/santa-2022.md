# RL / santa-2022

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 607 quality usable score 31

File: `RL/santa-2022/rank607_70pct/santa-2022-the-christmas-card-conundrum.ipynb`
Cells: 23 total, 21 code, 2 markdown

### Cell 0 markdown

```
**What are you trying to do in this notebook?**

My job is to determine the most optimal way to craft this year’s Christmas card, by selecting the most efficient path of both moving the robotic arm and changing the print color to craft this year’s image. Each link of the printer arm can be moved independently each step, but I'll also need to account for the time needed to change the printing color.

**Why are you trying it?**

To create a sequence of arm configurations covering every point on the image that minimizes the total movement of the arm and also the change in color from point to point.

The position of the arm is the sum of these displacement vectors and indicates the location of the tip of the arm. The base of the arm (the origin of the first vector) is at , which is the midpoint of the image.

The arm can be reconfigured step-by-step by rotating any or all of the links by 1 unit, incurring a total reconfiguration cost equal to the square root of the number of links changed. Additionally, it incurs a color cost equal to the sum of the absolute differences in the color components from one step to the next and multiplied by a scaling factor of 3.0.

The task is to find a sequence of configurations with positions at every point in the solution image having a minimal cost.
```

### Cell 1 code

```
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```

### Cell 3 code

```
df = pd.read_csv('/kaggle/input/santa-2022/image.csv')
```

### Cell 19 code

```
    flag = False
        if flag:
        flag = not flag
```

### Cell 21 code

```
submission.to_csv('submission.csv', index=False)
```

## 40pct rank 331 quality strong score 49

File: `RL/santa-2022/rank331_40pct/ddqn-lstm-reinforcement-learning.ipynb`
Cells: 15 total, 14 code, 1 markdown

### Cell 0 markdown

```
The problem being addressed is a variation of the Traveling Salesman Problem (TSP), which involves finding the shortest route that visits a given set of locations and returns to the starting point. This problem is being solved using a reinforcement learning approach, in which an agent learns to interact with its environment in order to maximize a reward. In this case, the reward is the minimization of the distance traveled. The agent learns through trial and error, receiving rewards for actions that lead to desirable outcomes and penalties for actions that do not. As the agent gathers more experience, it is able to improve its decision-making and eventually find the optimal solution to the problem.

The environment in which the problem is being solved is large, requiring a significant amount of time to train, with an estimated 300,000 steps(1 episode) taking approximately 24 hours to complete on a Kaggle GPU (P100). However, it is suggested that better results may be achieved by training for 10-20 episodes, though this is uncertain. The provided implementation of the problem is described as being easy to understand and well-commented.
```

### Cell 1 code

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
sub = pd.read_csv(p+'sample_submission.csv')
df_image = pd.read_csv(p+'image.csv')
```

### Cell 6 code

```
class CNNModel(nn.Module):
        super(CNNModel, self).__init__()
        x = torch.rand(1,4,257,257)
        out = torch.prod(torch.tensor(x.shape))
        torch.cuda.empty_cache()
        x = torch.rand(1,2,9)
        out = torch.prod(torch.tensor(x.shape))
        torch.cuda.empty_cache()
        x = torch.cat((x1, x2), dim=1)
```

### Cell 7 code

```
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # flag for using Double DQN variation
            image_state = image_state.to(dtype = torch.float32,device = device)
            config_state = config_state.to(dtype = torch.float32,device = device)
            action = torch.argmax(q_values).item()
        image_states_tensor = torch.cat(tuple([t[0] for t in transitions]), dim=0).to(device = agent.device,dtype = torch.float32)
        config_states_tensor = torch.cat(tuple([t[1] for t in transitions]), dim=0).to(device = agent.device,dtype = torch.float32)
        new_image_states_tensor = torch.cat(tuple([t[5] for t in transitions]), dim=0).to(device = agent.device,dtype = torch.float32)
        new_config_states_tensor = torch.cat(tuple([t[6] for t in transitions]), dim=0).to(device = agent.device,dtype = torch.float32)
        actions_tensor = torch.tensor(actions ,dtype = torch.int64).unsqueeze(-1)
        rewards_tensor = torch.tensor(rewards ,dtype = torch.float32).unsqueeze(-1)
        dones_tensor = torch.tensor(dones,dtype = torch.float32).unsqueeze(-1)
            next_actions_tensor = torch.argmax(q_eval_target,dim=0).unsqueeze(-1).data.cpu()
            action_q_values = torch.gather(input = q_next_target ,dim=0 ,index = actions_tensor)
        action_q_values = torch.gather(input = q_eval ,dim=0 ,index = actions_tensor)
        torch.cuda.empty_cache()
```

### Cell 8 code

```
    # convert the image array to a PyTorch tensor
    img = torch.tensor(img)
    config_ten = torch.tensor(config_arr)
```

### Cell 9 code

```
        # If the number of unique filled coordinates is greater than or equal to 66049 or the step count is greater than 250000, the done flag is set to True
```

### Cell 10 code

```
torch.cuda.empty_cache()
```

### Cell 11 code

```
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
online_net = CNNModel(n_actions,256).to(device)
target_net = CNNModel(n_actions,256).to(device)
```

### Cell 13 code

```
    image_states = torch.cat([image_state for _ in range(3)], dim=0).reshape(3,4,257,257)
    config_states = torch.cat([config_state for _ in range(3)], dim=0).reshape(3,2,9)
    score = 0
    prev_score = 0
        new_image_states = torch.cat((image_states[1:,:,:,:] ,new_image_state.view(-1,4,257,257)))[:,:,:,:].to(torch.float32)
        new_config_states = torch.cat((config_states[1:,:,:] ,new_config_state.view(-1,2,9)))[:,:,:].to(torch.float32)
        score += reward
            reward_buffer.append(score)
            print(f"step: {env.step_count}   reward: {score + prev_score}  epsilon: {agent.epsilon} Filled Points: {len(env.filled_coordinates)-prev_filled_points}")
            prev_score = -score
            torch.save(online_net, 'online_net.pt')
            torch.save(target_net, 'target_net.pt')
    print(score)
    torch.save(online_net, 'online_net.pt')
    torch.save(target_net, 'target_net.pt')
```

### Cell 14 code

```
import torch
torch.save(online_net, 'online_net.pt')
torch.save(target_net, 'target_net.pt')
# online_net = torch.load('online_net.pt')
# target_net = torch.load('target_net.pt')
```

## 20pct rank 186 quality usable score 43

File: `RL/santa-2022/rank186_20pct/santa-december.ipynb`
Cells: 36 total, 35 code, 1 markdown

### Cell 1 code

```
sub = pd.read_csv(p+'sample_submission.csv')
df_image = pd.read_csv(p+'image.csv')
```

### Cell 33 code

```
flag = False
        if not flag:
        flag = not flag
    flag = not flag
```

### Cell 35 code

```
submission.to_csv('submission.csv', index=False)
```

## 10pct rank 84 quality usable score 25

File: `RL/santa-2022/rank84_10pct/greedy-search-hill-climbing-random-explore.ipynb`
Cells: 18 total, 10 code, 8 markdown

### Cell 0 markdown

```
# Reference
I have referred to many existing notebooks. I would like to thank everyone.
- https://www.kaggle.com/code/ryanholbrook/getting-started-with-santa-2022
- https://www.kaggle.com/code/crodoc/82409-improved-baseline-santa-2022
- https://www.kaggle.com/code/nicupetridean/further-analysis-of-costs-points-ordering
- https://www.kaggle.com/code/snufkin77/further-point-order-improvements-tsp-start
- https://www.kaggle.com/code/elvenmonk/santa-2022-lower-bound-approximation-73078
```

### Cell 1 code

```
df_image = pd.read_csv(data_dir / 'image.csv')
```

### Cell 7 code

```
        flag = False # available go next step
                flag = True
        if not flag:
```

### Cell 13 code

```
        flag = False # available go next step
                flag = True
        if not flag:
```

## 1st rank 1 quality usable score 26

File: `RL/santa-2022/rank1_1st/lower-bound-using-minimum-spanning-tree.ipynb`
Cells: 6 total, 5 code, 1 markdown

### Cell 1 code

```
df_image = pd.read_csv('../input/santa-2022/image.csv')
```
