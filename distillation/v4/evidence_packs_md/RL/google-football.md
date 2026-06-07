# RL / google-football

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 783 quality usable score 33

File: `RL/google-football/rank783_70pct/gfootball-template-bot-modified.ipynb`
Cells: 6 total, 3 code, 3 markdown

### Cell 5 markdown

```
1. Go to [My Submissions](https://www.kaggle.com/c/football/submissions) to view your score and episodes being played.
```

## 40pct rank 410 quality strong score 55

File: `RL/google-football/rank410_40pct/gail-football.ipynb`
Cells: 24 total, 22 code, 2 markdown

### Cell 1 markdown

```
Attempt of high impact work from ( Ho & Ermon, 2016 ) known as Generative Adversarial Imitation Learning ( see https://arxiv.org/abs/1606.03476 ) for football. The implementation is adapted from this useful repository https://github.com/Khrylx/PyTorch-RL.
It can be useful also to see how to submit PyTorch agent with a compressed repository .tar.gz
Just run all this notebook and then compress the model "actor.pt" and "main.py" in a folder "submission.tar.gz". This is a valid format for the Kaggle competition.
```

### Cell 2 code

```
import torch
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
```

### Cell 3 code

```
tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros
    flat_params = torch.cat(params)
    flat_grad = torch.cat(grads)
    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)
    grads = torch.cat(out_grads)
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    advantages, returns = to_device(torch.device("cpu"), advantages, returns)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
            x = np.clip(x, -self.clip, self.clip)
```

### Cell 4 code

```
    torch.randn(pid)
            with torch.no_grad():
def merge_log(log_list):
        to_device(torch.device('cpu'), self.policy)
            log = merge_log(log_list)
        to_device(torch.device('cpu'), self.policy)
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
```

### Cell 6 code

```
import torch.nn as nn
import torch
            self.activation = torch.tanh
            self.activation = torch.relu
            self.activation = torch.sigmoid
            self.activation = torch.tanh
            self.activation = torch.relu
            self.activation = torch.sigmoid
        prob = torch.sigmoid(self.logic(x))
            self.activation = torch.tanh
            self.activation = torch.relu
            self.activation = torch.sigmoid
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        action_std = torch.exp(action_log_std)
        action = torch.normal(action_mean, action_std)
            self.activation = torch.tanh
            self.activation = torch.relu
            self.activation = torch.sigmoid
        action_prob = torch.softmax(self.action_head(x), dim=1)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))
```

### Cell 7 code

```
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.device('cpu')
```

### Cell 9 code

```
env = football_env.create_environment(env_name="11_vs_11_easy_stochastic", representation ="raw", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, render=False)
```

### Cell 11 code

```
torch.manual_seed(seed)
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=learning_rate)
```

### Cell 17 code

```
expert_states = torch.from_numpy(expert_states).to(dtype).to(torch.device("cpu"))
expert_actions = torch.from_numpy(expert_actions).to(dtype).to(torch.device("cpu"))
```

### Cell 19 code

```
torch.cat([expert_states, expert_actions], 1)
```

### Cell 20 code

```
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    with torch.no_grad():
```

### Cell 21 code

```
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(torch.device("cpu"))
    actions_one_hot = to_categorical(np.stack(batch.action),19)
    actions_one_hot = torch.from_numpy(actions_one_hot).to(dtype).to(torch.device("cpu"))
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(torch.device("cpu"))
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(torch.device("cpu"))
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(torch.device("cpu"))
    with torch.no_grad():
        expert_state_actions = torch.cat([expert_states, expert_actions], 1)
        g_o = discrim_net(torch.cat([states, actions_one_hot], 1))
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=torch.device("cpu"))) + \
            discrim_criterion(e_o, zeros((expert_states.shape[0], 1), device=torch.device("cpu")))
        perm = LongTensor(perm).to(torch.device("cpu"))
        discrim_net.to(torch.device('cpu'))
```

## 20pct rank 226 quality usable score 35

File: `RL/google-football/rank226_20pct/baseline-bot.ipynb`
Cells: 14 total, 10 code, 4 markdown

### Cell 0 markdown

```
# Tunable Baseline Bot
```

### Cell 1 code

```
### This is a fork of https://www.kaggle.com/david1013/tunable-baseline-bot


##  Any upvotes below to his original notebook.
```

### Cell 2 markdown

```
### Tunable Parameters
```

### Cell 3 code

```
%%writefile submission.py

## TUNABLE BASELINE BOT

# Tune Here:
SPRINT_RANGE = 0.6

SHOT_RANGE_X = 0.7
SHOT_RANGE_Y = 0.2

GOALIE_OUT = 0.2
LONG_SHOT_X = 0.4
LONG_SHOT_Y = 0.2
```

### Cell 4 markdown

```
### Install
```

### Cell 5 code

```
# Install:
# Kaggle environments.
!git clone --quiet https://github.com/Kaggle/kaggle-environments.git
!cd kaggle-environments && pip install -q .

# GFootball environment.
!apt-get -qq update -y
!apt-get -qq install -y libsdl2-gfx-dev libsdl2-ttf-dev

# Make sure that the Branch in git clone and in wget call matches !!
!git clone --quiet -b v2.3 https://github.com/google-research/football.git
!mkdir -p football/third_party/gfootball_engine/lib

!wget -q https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.3.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install -q .
```

### Cell 6 markdown

```
### Agent
```

### Cell 7 code

```
%%writefile -a submission.py

from kaggle_environments.envs.football.helpers import *
from math import sqrt

directions = [
[Action.TopLeft, Action.Top, Action.TopRight],
[Action.Left, Action.Idle, Action.Right],
[Action.BottomLeft, Action.Bottom, Action.BottomRight]]

dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)

enemyGoal = [1, 0]
GOALKEEPER = 0
```

## 10pct rank 114 quality usable score 43

File: `RL/google-football/rank114_10pct/game-visualization.ipynb`
Cells: 9 total, 7 code, 2 markdown

### Cell 3 code

```
  res["score"] = obs["score"]
```

### Cell 5 code

```
  global prev_score_a, prev_score_b
  score_a, score_b = obs["score"]
  match_info.set_text(f"{name_left} {score_a} : {score_b} {name_right}")
```

## 1st rank 2 quality strong score 54

File: `RL/google-football/rank2_1st/google-football-episode-scraper.ipynb`
Cells: 10 total, 9 code, 1 markdown

### Cell 2 code

```
MIN_FINAL_RATING = 1500 # top submission in a match must have reached this score
```

### Cell 6 code

```
teams_df.sort_values('publicLeaderboardRank', inplace = True)
```

### Cell 7 code

```
        teams_df.sort_values('publicLeaderboardRank', inplace = True)
    team_episodes['avg_score'] = -1;
        agent_scores = [a['updatedScore'] for a in agents if a['updatedScore'] is not None]
        team_episodes.loc[i, 'updatedScore'] = [a['updatedScore'] for a in agents if a['submission']['teamId'] == team_id][0]
        if len(agent_scores) > 0:
            team_episodes.loc[i, 'avg_score'] = np.mean(agent_scores)
        final_score = max( [r['updatedScore'] for r_idx, (r_index, r) in enumerate(sub_rows.iterrows())
        team_episodes.loc[sub_rows.index, 'final_score'] = final_score
    team_episodes.sort_values('avg_score', ascending = False, inplace=True)
```

### Cell 9 code

```
    team_df = team_df[  (MIN_FINAL_RATING is None or (team_df.final_score > MIN_FINAL_RATING))]
    print('   {} in score range from {} submissions'.format(len(team_df), len(team_df.submissionId.unique() ) ) )
```
