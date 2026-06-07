# RL / halite

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 800 quality usable score 29

File: `RL/halite/rank800_70pct/explore-exploit.ipynb`
Cells: 20 total, 5 code, 15 markdown

### Cell 0 code

```
from IPython.display import Image
Image("../input/image1/Picture1.png")
```

### Cell 1 markdown

```
## Table of Content

* [Credits](#tag1)


* [Idea](#tag2)


* [Helper functions](#tag3)


* [Execution](#tag4)


* [Improvement](#tag5)

```

### Cell 2 markdown

```
<a id='tag1'></a>
```

### Cell 3 markdown

```
## Credits to...
```

### Cell 4 markdown

```
* [Swarm Intelligence with SDK](https://www.kaggle.com/kwabenantim/swarm-intelligence-with-sdk) I used the Controller() class except a few changes in assigning the actions for the agents. I like its organized structure and the way it keeps track of cells to avoid collision is very smart. Please make sure to check it out!
* [Getting started with halite](https://www.kaggle.com/alexisbcook/getting-started-with-halite) I used functions from this notebook as well.
```

### Cell 5 markdown

```
<a id='tag2'></a>
```

### Cell 6 markdown

```
## Idea

Lots of algorithms in reinforcement learning are derived from the idea of **balancing between exploration and exploitation**(https://en.wikipedia.org/wiki/Multi-armed_bandit). In this game, there are advantages and distadvantages of both strategies:

* Exploration: the ship can look up the surrounding area and find the cells with larger number of halites, but it costs time to travel to the desired area, also it risks getting collision with other ships
* Exploitation: the ship choose to either stay at the current position to mine or move a little bit to the surround area, but it might find nothing by just looking around aimlessly

There are differeny ways to achieve the transition between exploration and exploitation, here I used a probability paramter EPSILON to decide which way to go. It is a bit similar with [epsilon-greedy algorithm](https://imaddabbura.github.io/post/epsilon-greedy-algorithm/)
```

### Cell 7 markdown

```
<a id='tag3'></a>
```

## 40pct rank 445 quality usable score 37

File: `RL/halite/rank445_40pct/halite-result-visualization-from-leaderboard.ipynb`
Cells: 28 total, 22 code, 6 markdown

### Cell 10 code

```
df_merged = pd.concat([df0,df1,df2,df3])
```

### Cell 11 code

```
df_merged['total_halite'] = df_merged['halite'] + df_merged['cargo']
df_merged['cargo_average']  = df_merged['cargo'] / df_merged['ships']
df_merged['cargo_percentage'] = df_merged['cargo'] / df_merged['total_halite']
```

### Cell 12 code

```
df_merged
```

### Cell 14 code

```
sns.barplot(data=df_merged[df_merged['step']==last_step],x='player',y='halite',ci=None)
```

### Cell 15 code

```
sns.barplot(data=df_merged,x='player',y='halite',ci=None)
```

### Cell 16 code

```
sns.barplot(data=df_merged[df_merged['step']==last_step],x='player',y='cargo',ci=None)
```

### Cell 17 code

```
sns.lineplot(data=df_merged,x='step',y='halite' ,hue='player')
```

### Cell 18 code

```
sns.lineplot(data=df_merged,x='step',y='halite' ,hue='player')
```

### Cell 19 code

```
sns.lineplot(data=df_merged,x='step',y='cargo', hue='player')
```

### Cell 20 code

```
sns.lineplot(data=df_merged,x='step',y='cargo', hue='player')
```

### Cell 21 code

```
sns.lineplot(data=df_merged,x='step',y='total_halite', hue='player')
```

### Cell 22 code

```
sns.lineplot(data=df_merged,x='step',y='cargo_percentage', hue='player')
```

## 20pct rank 225 quality usable score 41

File: `RL/halite/rank225_20pct/halite-bot-aggressive-bot-starter-code.ipynb`
Cells: 4 total, 2 code, 2 markdown

### Cell 0 markdown

```
This is the starter code for a strategy that will be focused on attacking other ships with less halite.  Please provide any feedback/suggestion, as they would be much appreciated.
So far, it's score has been pretty low, but I think that the functions are still useful and that this strategy still has potential.  I will continue updating the code as I go.
```

### Cell 1 code

```
# Code to attack closest ship
def attackShip(ship, other_ships, size):
    attack = nearestObject(ship.position, other_ships, size)
    pos, halite = attack
# Code to attack shipyard: write if's in agent
#def attackShipyard(ship, other_shipyards, size):
#    attack = nearestObject(ship.position, other_shipyards, size)
#    pos, halite = attack
                ship_states[ship.id] = "ATTACK"
            if ship_states[ship.id] == "ATTACK":
                direction = attackShip(ship, other_ships, size)
```

## 10pct rank 118 quality strong score 55

File: `RL/halite/rank118_10pct/pytorch-starter.ipynb`
Cells: 17 total, 14 code, 3 markdown

### Cell 4 code

```
import torch
model.load_state_dict(torch.load(buffer, torch.device("cuda" if torch.cuda.is_available() else "cpu")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Cell 5 code

```
torch.no_grad();
torch.set_num_threads(os.cpu_count())
```

### Cell 6 code

```
    input_stack = getInputStack(step, halite, ship, base, cargo, ph, new_var, prior_actions, action_map,
    input_stack   = torch.as_tensor(input_stack).unsqueeze(0).to(device)
    output = model(input_stack)
    base_prediction = torch.sigmoid( policy_output[:, -1, :] ).detach().numpy()
    # score matrix -- can only pick valid actions
        ship_ranked_actions = np.zeros((6,), dtype = np.float32)
        for rank in range(0, np.sum(ship_pred_actions > 1e-6) ):
                    ship_ranked_actions[action] = 6 - rank + raw_ship_pred_actions[action];
        print(list(np.round(ship_ranked_actions, 1)))
        C[ship_idx, x + 21*y] = ship_ranked_actions[0]
        C[ship_idx, x + 21*c(y - 1)] = ship_ranked_actions[1]
        C[ship_idx, c(x + 1) + 21*y] = ship_ranked_actions[2]
        C[ship_idx, x + 21*c(y + 1)] = ship_ranked_actions[3]
        C[ship_idx, c(x - 1) + 21*y] = ship_ranked_actions[4]
            C[ship_idx, 21*21 + ship_idx]= ship_ranked_actions[5] # conversion doesn't use any squares
 #     print("{:.0f} ms - rankings".format((datetime.datetime.now() - start_time).microseconds // 1e3))
```

## 1st rank 2 quality strong score 48

File: `RL/halite/rank2_1st/halite-iv-count-win-or-lose-for-each-team.ipynb`
Cells: 8 total, 8 code, 0 markdown

### Cell 0 code

```
# (When both are specified, score_threshold_from_best_in_team has more priority than number_of_best_agents.)
# I don't know about the most suitable threshold to describe real performance.
score_threshold_from_best_in_team = None
# Ranks of opponent teams
#       to the number of episodes of teams within team_rank_range.
team_rank_range = range(1, 13)
```

### Cell 5 code

```
def get_target_submission_ids_impl(team_k, submission_id_to_team_id, max_scores, update):
    global score_threshold_from_best_in_team
        # maybe this sort key is wrong to get latest scores
            continue  # skip validation
            updated_score_k_e = agent_k_e_p['updatedScore']
            if updated_score_k_e is not None:
                submission_id_candidates_k[submission_id_k_e] = updated_score_k_e
    max_scores[team_id_k] = sorted_submission_id_candidates_k[0][1]
    for submission_id_k_s, score_k_s in sorted_submission_id_candidates_k:
        s = f'team_id={team_id_k} submission_id={submission_id_k_s} score={score_k_s}'
        if score_threshold_from_best_in_team:
            if score_k_s < max_scores[team_id_k] - score_threshold_from_best_in_team:
    teams.sort(key=lambda x: x['publicLeaderboardRank'] or 1e9)
    max_scores = {}
    for rank in team_rank_range:
        k = rank - 1  # rank is 1-indexed
        get_target_submission_ids_impl(team_k, submission_id_to_team_id, max_scores, update)
    if your_team['publicLeaderboardRank'] not in team_rank_range:
        get_target_submission_ids_impl(your_team, submission_id_to_team_id, max_scores, update)
    return submission_id_to_team_id, max_scores
```

### Cell 6 code

```
# This method may take a long time if team_rank_range is too wide
submission_id_to_team_id, max_scores = get_target_submission_ids(list_episodes)
```

### Cell 7 code

```
for team_id in max_scores.keys():
    match_results[team_id] = {'team_name': team_name, 'max_score': round(max_scores[team_id], 1), 'win': 0, 'draw': 0, 'lose': 0}
```
