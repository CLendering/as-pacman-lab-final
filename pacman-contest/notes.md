




One bug I had in my particle filter was that I did this to get the legal actions for a position (x, y):

```
from contest.game import Directions, Configuration, Actions
dummy_config = Configuration((x, y), 'North')
possible_actions = Actions.get_possible_actions(dummy_config, self.walls)
```

But using 'North' for the dummy_config would cause `Actions.get_possible_actions` to return just ['North'] in some cases and disregard the walls! If you use 'Stop' for the dummy_config, it works as expected.

Just so you know and don't get fucked by the same bug haha


# particle filter todos

check out display_distributions_over_positions

use systematic resample instead of multinomial in __resample_particles


When pacman is eaten, he respawns immediately and loses the food he was carrying.

# improving distance reading
- actually kalman filter might not be best idea bc it assumes linearity & gaussianity etc. maybe non-linear filter is better? read into particle filter?
- or just use a neural net/lstm to train a better distance estimator
- use wall positions to plausibilize possible enemy positions calculated from own agent position + distance estimate
- triangulate where enemies might be by using enemy position estimates of both our own agents

- particle filter:     # - enemies move slower when they're scared ghosts


# kalman filter
- update when we get an exact position
- draw some plots comparing different adjustments
- tune the parameters of the filter


# Ideas
- confuse enemies by stacking both agents on top of each other. but I guess this would only confuse agents who parse the map or use the noisy distance estimates. maybe you 

- check different layouts

- dieing isn't that bad as long if you bring food closer to home (bc you can get food from far enemy territory and bring it closer (it drops near you when you die))

- save observations to make prediction clearer (Kalman filter?). Noise values go from -6 to +6, so they could cancel out if we make multiple observations. but we have to remember that an observation t timesteps ago can have moved up to +t/-t further, so trust them less.

I have to check of what kind the distances are. Are they Manhattan Distances or Maze Distances (see class Distancer)? --> they are manhattan distances, but you can use CaptureAgent.distancer to calculate maze distances as well. these take the layout of the map into account. 

"an agent always gets a noisy distance reading for each agent on the board, which can be used to approximately locate unobserved opponents"

one could also constrain the estimated distances to be plausible with the given map layout

if the agents can communicate, we could do sensor fusion. ocasionally an agent can get the exact position of an enemy with get_agent_position, he should tell that to the other agent as well.

`game_state.get_agent_position()` seems to always give you a noisy estimate. at least it already gives you a noisy estimate for your team ally when you start (where you're literally 1 cell apart). it even gives you a noisy distance estimate to yourself lol!

in def register_initial_state you can call game_state.get_agent_position(i) with the indices of the enemies and you will get their exact starting position. -> use to initialize kalman filter

you can always get the position of your teammate.

- you can use keyboard to control agents. However, something's broken in def keys_pressed(d_o_e=lambda arg: _root_window...) because _root_window is always null even though it was set before. using tkinter._default_root instead is a hacky workaround, but that just stops the crash; you still can't control the agents.

- kill agents who carry a lot of food

- use power capsule to steal a lot of food. we know that the opposing team's ghosts become scared for the next 40 moves, so we can plan ahead and do planning with that in mind. it doesn't make sense to eat a ghost because they respawn immediately.

- if I'm a scared ghost, it makes sense for me to die rather than to wait for 40 moves. If I only have to wait for a short time I would rather not die. I could also move to the other side so I become a pacman instead of a scared ghost.

- max 300 moves per agent - play riskier if less time and we're losing / play safer if less time and we're winning w/ good margin

# Map legend
Red team is always on the left, blue team is always on the right. `create_team` gets parameter `is_red` to know which side we play on.

`%`: Wall  
` `: clear  
`.`: food
`o`: capsule (power pellet)  
`G`: ghost
`^`: pacman facing down
`>`: pacman facing left
`<`: pacman facing right
`v`: pacman facing up


```
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   %. %.%.%       %     %.%.% %
% % %% G     %%  %   %%%   %.% %
% % %. % %%%    %%%% .%..% % % %
% % %% % ..% %   %   %%%%% % % %
% %    %%%%% %%%   %%%.% o % % %
% %% % ..%.  % %%%       %   % %
% %. %%.%%%%  ^     %.%%%%  %% %
% %%  %%%%.%     v  %%%%.%% .% %
% %   %       %%% %  .%.. % %% %
% % % o %.%%%   %%% %%%%%    % %
% % % %%%%%   %   % %.. % %% % %
% % % %..%. %%%%    %%% % .% % %
% %.%   %%%   %  %%       %% % %
% %.%.%     %       %.%.% .%   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Score: 0
```

# Functions

## Methods of CaptureAgent
See `class CaptureAgent` for documentation
    agentsOnTeam
    choose_action
    debug_clear
    debug_draw
    display
    display_distributions_over_positions
    distancer
    evaluate
    final
    get_action
    get_capsules
    get_capsules_you_are_defending
    get_current_observation
    get_features
    get_food
    get_food_you_are_defending
    get_maze_distance
    get_opponents
    get_previous_observation
    get_score
    get_successor
    get_team
    get_weights
    index
    observationHistory
    observation_function
    red
    register_initial_state
    register_team
    start
    timeForComputing

## Methods of game_state
See `class GameState` for documentation  
    agent_distances
    blue_team
    data
    deep_copy
    generate_successor
    get_agent_distances
    get_agent_position
    get_agent_state
    get_blue_capsules
    get_blue_food
    get_blue_team_indices
    get_capsules
    get_distance_prob
    get_initial_agent_position
    get_legal_actions
    get_num_agents
    get_red_capsules
    get_red_food
    get_red_team_indices
    get_score
    get_walls
    has_food
    has_wall
    initialize
    is_on_red_team
    is_over
    is_red
    make_observation
    red_team
    teams
