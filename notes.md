so I'm a little confused about how/where we're defining the sweep.


HOW the rewards are assessed (algorithms for each award) are defined in gym_env.py
VALUES for the rewards are defined in reward_sweep_GPU.py ?
but it seems like the reward_sweep_GPU.py isn't actually sweeping with the reward grid like I was previously in reward_sweep_no_stopping.py

so we have various reward CONFIGURATIONS

ENVIRONMENT SETUP
sounds like regardless of how many reward configurations we have, we set up 32 parallel environments
each of those 32 environments has some reward weights... not sure how that relates to the reward configurations from above.

DQN TRAINING "DEEP Q LEARNING" this is "deep neural network" training.
this uses the environments from above
here we define the 
number of timesteps e.g. 100,000
large buffer = 200,000
large batch size = 1024
device = "cuda" (GPU)
then it runs through hundreds of thousands of actions across all ENVIRONMENTS (remember there are 32 environments in parallel)
By doing this it improves its "Q-values"

EVALUATION


so previously, we were tallying up awards after every move. e.g. if you moved max tile out of corner you got a big negative reward.
sounds like this is only rewarding after the training is finished?

after training, the "trained model" plays 128 games aka episodes
for each game, a fresh game env is created, and the same reward settings are used. I think this means the same "configuration" e.g. tile_bonus = 1.5, max_out_of_corner = -5 etc
the game runs until it ends, OR it hits 1000 moves which is a lot of moves. 
total reward is recorded.

so I guess somehow we do "training" but then we don't assign any rewards until AFTER the training is done... it's like the rewards are used more like a final exam than continuous reinforcement.

so then after the 128 episodes, we record the average reward and the stddev of the rewards (to see how consistent it was)

so I guess we run 128 games aka episodes for each configuration

but then it prints out a summary

=====================================

REVISED UNDERSTANDING

ok so we basically hard-code some configurations instead of creating a grid
so like less than 10 configs. fine for now
then for EACH of those configs, we create 32 parallel environments to train on.
EACH of those 32 environments for that config runs DQN for 100k "timesteps" where one timestep is just one up/down/left/right action for our use case.
So to clarify, for EACH config, we create 32 parallel environments. And for a given config those 32 environments share 100k steps total. In other words, for each config we have a total of 100k up/down/left/right actions, each of which is immediately assessed for rewards.

so that's the training phase. during the training phase, the model was changing its behavior as it was learning. now it has "finished" learning and its behavior is mature.
THEN we go into the evaluation phase, play 128 games (for statistical significance) for each configuration, to assess how well each configuration was able to train the model. "using the same reward weights as training" --> this is confusing to me, because if this is a "test", we wouldn't actually be using the rewards. we would be assessing how well we did on completely different metrics like max tile achieved and "cornerness" aka how well we kept the max tile in the corner. If we were just using reward weights, it would be hard to compare config to config because the weights are different. so one model with big weights could "score" higher on the test even though it "performed" worse aka didn't produce high max tiles.

follow-up thought/question - My PC is new and beastly (20 cores, 16gb VRAM). When I ran this, it completed in only 77.3 seconds. While it was running, I had 1 cpu working at 100% and GPU was at like 10% with like <2gb VRAM used. Could we do this BETTER and/or FASTER or otherwise take more advantage of my hardware?


===================

NEXT STEPS

- play a game counting moves, I think 1000 move max might be too low. Get a feel for how many moves are required to hit 256, 512, 1024 (extrapolate from there)

for the evaluation phase, create a neutral "exam" so that reward weights don't bias the measurement of efficacy.
some metrics might be:
- max tile achieved
- total game score which is defined in game.py (note this is NOT the same as the reward tally)
- monotonicity e.g. making the snake pattern where highest tile is in bottom left, above that is smaller etc then it "snakes" around where the next column decreases from top to bottom

Improve parallelization ideas:
- use SubprocVecEnv instead of DummyVecEnv #DONE
- use batched inference or multi-threaded eval e.g. ThreadPoolExecutor

Improve reward function evaluation
- max tile per episode
- cornerness (how well we're keeping max tile in corner)
- game score (from game.py, separate from reward score)
- other metrics (total merges, monotonicity)

Store those metrics in a list and compute mean & stddev for each config
Maybe save in json or CSV for plotting


=========================
Me trying to understand parameters and how this all works:

we are basically training two separate but related things: policy update and critic update.
policy is WHAT we do - given a board, this is the up/down/left/right decision
critic is how a given board is evaluated - for any given board position, we try to estimate the resultant "goodness" of the board after we make a given move. e.g. if we have a given board and move LEFT, what is the single-number "value" of that move? The better the critic model, the closer our prediction matches the result 

Our parameters and my understanding:
-subproc # parallelization method - subproc=actual, vs dummy
--num-envs 64 # how many environments we're running in parallel. this is so we aren't doing all runs sequentially.
--total-steps 180000000 # total runtime across all environments. this would be like the total number of up/down/left/right across the entire run
--n-steps 4096 # ?
--batch-size 131072 # ? something to do with how we divide epochs. not sure what an epoch is in our context  
--arch 1024,1024 # no idea 
--lr 5e-5 # learning rate. weight for how strongly we update our learning based on our findings. Learning includes policy update and critic update. I think this weight applies to both?
--n-epochs 20 # relates to batch size... although I thought total-steps = n_epochs*batch-size. but that math doesn't add up and also it would be overdefined since we have three variables and are defining all three.
--entropy-start 0.03 
--entropy-end 0.01 
--clip-range 0.1 
--gamma 0.995 
--vecnorm 
--episode-steps 3000 
--curriculum # curriculum enable
--curr-target-start 128
--curr-promote 0.45 
--curr-window 192 
--curr-bonus 2.5 
--expert-assist # expert assist enable
--expert-prob 0.01 
--expert-depth 1 
--expert-k 3 
--tb runs/god_mode_confirm_s2_180M_true_s2 
--seed 2 # seed used for "random" number generation. I THINK this means that if you start a new game and make the same moves, your games will be deterministic e.g. have the same result.
--eval-episodes 512 # test at the end? I didn't think we were doing this anymore
--results god_mode_results_confirm_s2_180M_true_s2.json 
--save runs/god_mode_confirm_s2_180M_true_s2/final_model




1. regarding naming, looks like we're using "s2" to refer to both the series and the seed.
e.g. in these filenames:
god_mode_confirm_s2_180M_true_s2/PPO_1 
 god_mode_confirm_s2_180M_true_s3/PPO_1 

from now on let's explicitly write "seed2" or "seed3" so we don't get it mixed up with the series. Or let's just omit the "series" altogether, or if we want to keep the series maybe give it a "version" like v1 v2 etc

2. so sometimes we will use an expert mvoe (at random, as determined by expert-prob e.g. .01 will make 1% of moves be an expert move). 
3. I'm skeptical of the heuristic evaluator; I don't recall seeing this tested thoroughly. Does this update as the model trains? I'd like to see examples of this in action. Perhaps you can give a few board examples and show me how they would be evaluated for an expert move.
4. 
