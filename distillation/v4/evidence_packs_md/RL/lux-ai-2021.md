# RL / lux-ai-2021

Buckets: `70pct -> 40pct -> 20pct -> 10pct -> 1st`

## 70pct rank 819 quality strong score 61

File: `RL/lux-ai-2021/rank819_70pct/delux-ai-zero.ipynb`
Cells: 30 total, 17 code, 13 markdown

### Cell 1 markdown

```
## Notes about MuZero

Inputs:
- current and past game states

Model Functions:
- Representation function h(observation) -> hidden state
- Prediction function f(hidden state) -> policy and value
- Dynamic function g(hidden state, action) -> reward and next hidden state

Things to predict:

- policy
- value
- reward
```

### Cell 3 code

```
import tensorflow as tf
```

### Cell 10 code

```
class MultiHeadAttention(tf.keras.layers.Layer):
        self.wq = tf.keras.layers.Dense(d_input)
        self.wk = tf.keras.layers.Dense(d_input)
        self.wv = tf.keras.layers.Dense(d_input)
        self.dense = tf.keras.layers.Dense(d_input, activation=tf.nn.elu)
```

### Cell 11 code

```
class UnitEncoder(tf.keras.layers.Layer):
        self.conv = tf.keras.layers.Conv1D(16, 1)
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.elu)
```

### Cell 14 code

```
class SpatialEncoder(tf.keras.layers.Layer):
        self.project = tf.keras.layers.Conv2D(32, 1, activation=tf.nn.elu)
        self.conv1 = tf.keras.layers.Conv2D(4, 5, activation=tf.nn.elu, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(4, 5, activation=tf.nn.elu, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(2, 5, activation=tf.nn.elu, padding='same')
        self.dense = tf.keras.layers.Dense(64, activation=tf.nn.elu)
        self.flatten = tf.keras.layers.Flatten()
        self.conv_scatter = tf.keras.layers.Conv1D(4, 1, activation=tf.nn.elu)
```

### Cell 17 code

```
class ScalarEncoder(tf.keras.layers.Layer):
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(16, activation=tf.nn.elu)
```

### Cell 19 markdown

```
# Model Building
```

### Cell 21 code

```
    def get_ucb_score(self, parent_node):
        ucb_score = [0]*32
                ucb_score[i] = c_1 + np.log((parent_node.visit_count + c_2 + 1)/c_2)
                ucb_score[i] *= parent_node.P[i].numpy() * np.sqrt(parent_node.visit_count)
                ucb_score[i] = c_1 + np.log((parent_node.visit_count + c_2 + 1)/c_2)
                ucb_score[i] *= parent_node.P[i].numpy() * (np.sqrt(parent_node.visit_count))/(1+parent_node.children[i].visit_count)
                ucb_score[i] += parent_node.children[i].Q
        return tf.constant(ucb_score)
            ucb_score = self.get_ucb_score(selected_node)
            #print(ucb_score)
            selected_policy_index = np.argmax(ucb_score)
```

### Cell 22 markdown

```
## Model class
Representation model

remove residual block and use conv for now

Inputs: game state map
Outputs: list of actions

since the muzero is designed for singe agent game, we have to improvise a bit.

Methods:
- Self play
- Learn from replay buffer
```

### Cell 23 code

```
class Delux(tf.keras.Model):
        self.loss_cce = tf.keras.losses.CategoricalCrossentropy()
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.05)
        f_input = tf.keras.Input(shape=(96))
        f = tf.keras.layers.Dense(48, activation=tf.nn.elu)(f_input)
        v = tf.keras.layers.Dense(8, activation=tf.nn.elu)(f)
        v = tf.keras.layers.Dense(1)(v)
        p = tf.keras.layers.Dense(32, activation=tf.nn.elu)(f)
        p = tf.keras.layers.Dense(32, activation=tf.nn.softmax)(p)
        self.Prediction = tf.keras.Model([f_input], [p, v], name='Prediction Model')
        g_hidden_state = tf.keras.Input(shape=(96))
        g_hidden_actions = tf.keras.Input(shape=(32))
        g = tf.keras.layers.Dense(96, activation=tf.nn.elu)(g_input)
        g = tf.keras.layers.Dense(96, activation=tf.nn.elu)(g_input)
        r = tf.keras.layers.Dense(32, activation=tf.nn.elu)(g)
        r = tf.keras.layers.Dense(8, activation=tf.nn.elu)(r)
        r = tf.keras.layers.Dense(1)(r)
        self.Dynamic = tf.keras.Model([g_hidden_state, g_hidden_actions], [g, r], name='Dynamic Model')
        h_input = tf.keras.Input(shape=(192))
        h = tf.keras.layers.Dense(96, activation=tf.nn.elu)(h_input)
        h = tf.keras.layers.Dense(96, activation=tf.nn.elu)(h)
        h = tf.keras.layers.Dense(96, activation=tf.nn.elu)(h)
        self.Representation = tf.keras.Model(h_input, h, name='Representation Model')
        e_input = tf.keras.Input(shape=(23))
        e = tf.keras.layers.Dense(32, activation=tf.nn.elu)(e_input)
        e = tf.keras.layers.Dense(32, activation=tf.nn.elu)(e)
        e = tf.keras.layers.Dense(32, activation=tf.nn.softmax)(e)
        self.policyEncoder = tf.keras.Model(e_input, e, name='Policy Encode
...[truncated]
```

### Cell 27 code

```
            player_score = game_state.players[0].city_tile_count * 3 + len(game_state.players[0].units) + game_state.players[0].research_points / 100
            opponent_score = game_state.players[1].city_tile_count * 3 + len(game_state.players[1].units) + game_state.players[1].research_points / 100
            print(f"Episode {i}\nPlayer score: {player_score}\nOpponent score: {opponent_score}\nSurvived turn: {obs.step}")
```

## 40pct rank 438 quality strong score 61

File: `RL/lux-ai-2021/rank438_40pct/lux-ai-deep-learning-imitate-the-strategy.ipynb`
Cells: 35 total, 23 code, 12 markdown

### Cell 0 markdown

```
- [ ] custom loss and metrics
```

### Cell 5 markdown

```
The inputs of the model will be the map of the features, The output also will be a map of actions as matrix.
```

### Cell 9 code

```
The agent which was uploaded from Github follow a good strategy and save automatically the history in snapshots folder.
```

### Cell 18 code

```
    # stacking all in one array
    E = np.dstack([M,U_opponent,U_player,C_opponent,C_player])
```

### Cell 20 code

```
from tensorflow.keras.utils import  plot_model ,Sequence
    'Generates data for Keras'
```

### Cell 22 code

```
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_hub as hub
from tensorflow.keras import backend as K
    is_unit = tf.keras.backend.max(y_units_true,axis = -1)
    is_city = tf.keras.backend.max(y_cities_true,axis = -1)
    y_units_pred*= K.stack([is_unit]*6, axis=-1)
    y_cities_pred*= K.stack([is_city]*2, axis=-1)
    is_unit = tf.keras.backend.max(y_units_true,axis = -1)
    y_units_pred*= K.stack([is_unit]*6, axis=-1)
    is_city = tf.keras.backend.max(y_cities_true,axis = -1)
    y_cities_pred*= K.stack([is_city]*2, axis=-1)
```

### Cell 23 markdown

```
# Model
```

### Cell 24 code

```
# from keras import  models
```

### Cell 25 code

```
# from keras.models import Model
```

### Cell 26 code

```
# autoencoder.fit_generator(train_generator, steps_per_epoch=5,epochs= 10,validation_data=val_generator,validation_steps=5)
```

### Cell 27 code

```
from tensorflow.keras import activations
    inputs = keras.Input(shape=get_inputs(game_state).shape,name = 'The game map')
    model = keras.Model(inputs = inputs, outputs = output)
    opt = keras.optimizers.Adam(learning_rate=0.000001)
    model.compile(optimizer= "adam", loss= custom_mean_squared_error ,metrics = ["accuracy"])
```

### Cell 28 code

```
tf.keras.utils.plot_model(
    rankdir="TB",
```

## 20pct rank 272 quality usable score 43

File: `RL/lux-ai-2021/rank272_20pct/lux-ai-rule-based-agent.ipynb`
Cells: 36 total, 22 code, 14 markdown

### Cell 8 code

```
        self.attached_player_city_tiles_pos = set()
        self.attached_opponent_city_tiles_pos = set()
        new_cluster.attached_player_city_tiles_pos = set.union(self.attached_player_city_tiles_pos,
                                                               other.attached_player_city_tiles_pos)
        new_cluster.attached_opponent_city_tiles_pos = set.union(self.attached_opponent_city_tiles_pos,
                                                                 other.attached_opponent_city_tiles_pos)
                self.attached_player_city_tiles_pos.add(pos)
                self.attached_opponent_city_tiles_pos.add(pos)
        if len(self.attached_player_city_tiles_pos) > 0:
            if len(self.attached_opponent_city_tiles_pos) > 0:
            if len(self.attached_opponent_city_tiles_pos) > 0:
```

### Cell 15 code

```
        Builds a grid of possible expansion spots with specific expansion values depending on the amount of attached
                    # exclude all expansion positions if they are not attached to wood tiles.
                max_num_expansions = cluster.size - len(cluster.attached_player_city_tiles_pos)
                # get attached district mayors:
                attached_district_majors = set()
                    for pos in cluster.attached_player_city_tiles_pos:
                            attached_district_majors.add(dist_major)
                    for att_dist_mayor in attached_district_majors:
                    Attached cities. But we need to protect this gate!
```

### Cell 17 code

```
                                          if st not in cluster.attached_player_city_tiles_pos]
```

### Cell 19 code

```
        c) And we ensure that expansion spots ar not only attached to coal or uranium tiles. Otherwise they might be
            # use only expansion positions that are not attached to a wood cluster and that are not closing a cyrcle.
                # use only expansion positions that are not attached to a wood cluster
```

### Cell 21 code

```
                        # add units to harvesters if they are not part of an coal cluster that is under attack
                if len(cluster.attached_player_city_tiles_pos) >= 2:
                                            and (len(c.attached_player_city_tiles_pos) < 2)
                                            and (len(c.attached_player_city_tiles_pos) < 2)
        First of all we need to know in which state we are. In terms of night and day shift.
                if (len(cluster.attached_player_city_tiles_pos) == 2) and (cluster.num_surrounding_units == 2):
                elif (len(cluster.attached_player_city_tiles_pos) == 3) and (cluster.num_surrounding_units > 2):
                elif (len(cluster.attached_player_city_tiles_pos) == 4) and (cluster.num_surrounding_units > 2):
                num_city_tiles_to_support = len(cluster.attached_player_city_tiles_pos)
                for tile_pos in cluster.attached_player_city_tiles_pos:
            if (len(cluster.attached_player_city_tiles_pos) == 2) and (cluster.num_surrounding_units == 2):
            elif (len(cluster.attached_player_city_tiles_pos) == 3) and (cluster.num_surrounding_units > 2):
            elif (len(cluster.attached_player_city_tiles_pos) >= 3) and (cluster.num_surrounding_units > 2):
                    and (len(spot.other_cluster.attached_player_city_tiles_pos) == 0):
```

## 10pct rank 117 quality strong score 73

File: `RL/lux-ai-2021/rank117_10pct/fast-il-in-keras.ipynb`
Cells: 20 total, 15 code, 5 markdown

### Cell 2 code

```
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras import layers
```

### Cell 9 code

```
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
    b[x_shift:32 - x_shift, y_shift:32 - y_shift, 19] = 1
            shift = (32 - size) // 2
            x = unit_pos[0] + shift
            y = unit_pos[1] + shift
```

### Cell 12 code

```
    metrics=['accuracy']
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
```

### Cell 14 code

```
    validation_data=val_seq,
```

### Cell 15 code

```
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
```

### Cell 17 code

```
import tensorflow as tf
model = tf.keras.models.load_model(f'{path}/model.h5')
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
    b[x_shift:32 - x_shift, y_shift:32 - y_shift, 19] = 1
    for label in np.argsort(policy)[::-1]:
```

## 1st rank 5 quality usable score 42

File: `RL/lux-ai-2021/rank5_1st/select-agents-for-downloading-matches.ipynb`
Cells: 50 total, 29 code, 21 markdown

### Cell 1 markdown

```
Other scraping strategies can be implemented, but not here. Like download max X matches per submission or per team per day, or ignore certain teams or ignore where some scores < X, or only download some teams.
```

### Cell 7 code

```
episodes_df = pd.read_csv(META + "Episodes.csv")
epagents_df = pd.read_csv(META + "EpisodeAgents.csv")
```

### Cell 10 code

```
submission_id_to_episode_id = epagents_df.groupby('SubmissionId').head()[['SubmissionId', 'EpisodeId']]
```

### Cell 13 code

```
# teams = pd.read_csv(META + "Teams.csv")
```

### Cell 15 code

```
#submissions = pd.read_csv(META + "Submissions.csv")
```

### Cell 27 code

```
leaderboard = epagents_df.groupby('SubmissionId').tail(1)
leaderboard = leaderboard[~leaderboard.UpdatedScore.isna()]
leaderboard = leaderboard.sort_values('UpdatedScore', ascending=False)
```

### Cell 28 markdown

```
So we have 8k agents with a numeric score. Let's see the distribution of scores.
```

### Cell 29 code

```
plt.hist(leaderboard.UpdatedScore, bins=1000, log=True, cumulative=-1, histtype='stepfilled');
```

### Cell 30 markdown

```
We can see that there are around 250 agents above 1500 score.
```

### Cell 35 code

```
    if verbose: print('Scrolling results...')
    scrolling_element = browser.find_element(
        browser.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrolling_element)
```

### Cell 37 markdown

```
To download the files I need the episode id. I only want to download matches from the agents with the highest ranking.
Thus I'm going to update `epagents_df` to include the final ranking and later remove agents with low score.
```

### Cell 38 code

```
submission_id_to_final_scores = {key: value for key, value in zip(leaderboard.SubmissionId, leaderboard.UpdatedScore)}
epagents_df['FinalScore'] = epagents_df['SubmissionId'].apply(lambda x: submission_id_to_final_scores.get(x, -100))
```
