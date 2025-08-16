Cursor explanation:

Here’s a point-by-point check and plain-English clarifications.
1) Episode definition
Correct. One episode = one game. It ends when:
no legal moves (game over), or
time limit reached (truncated), or
curriculum target reached (early stop).
2) Curriculum + terminal bonus
Correct. We pick a target tile (e.g., 128). If you reach it, the episode ends and we add a small fixed bonus.
Terminal bonus: a constant you set via --curr-bonus (e.g., 2.5). We simply add that number to the reward when the target is reached.
3) Promotion logic (raising the target)
Correct. We track success rate over a sliding window of episodes (e.g., 192). If rate ≥ threshold (e.g., 0.45), we promote the target (128→256→512…).
Benefit: it rewards the “good prefix” up to the target and avoids reinforcing any messy play that might follow if the episode continued.
4) “Mix of current and next targets”
Each new episode samples which target to use:
Usually the current target (easier),
Sometimes the next target (harder).
This keeps some exposure to the next difficulty so the policy doesn’t overfit to just the current goal and forgets transitions.
5) n_envs and n_steps
We run many environments in parallel (n_envs). Each environment keeps running episodes (resetting as needed) until it has produced n_steps transitions. Across n_envs, that gives n_envs × n_steps samples per update.
n_steps is “steps collected per env before we do a policy update,” not a cap on episode length (the cap is --episode-steps).
6) “Compute advantages from rewards and the value head”
The critic (value head) estimates “how good this state is.” Advantage ≈ (what actually happened) − (what was expected).
If an action led to better-than-expected outcomes, its advantage is positive (reinforce it). If worse, negative (discourage it).
7) Policy update, value loss, entropy bonus
Policy update: we nudge the policy toward actions with positive advantage, but with PPO’s “clipping” to keep changes small/safe.
Value loss: we also train the critic to better predict “how good this state is.”
Entropy bonus: we reward the policy for staying a bit random early on (exploration), then decay it over time.
8) VecNormalize, expectimax, DAGGER (quick)
VecNormalize: normalizes observations across all envs so scales are stable for learning.
Expectimax: a short lookahead search that evaluates moves by averaging over random tile spawns; acts like a strong heuristic.
DAGGER-like assist: during training, with small probability we replace the policy’s action with the expectimax action. That exposes the policy to good mid/late-game states without a full imitation loop.
9) Evaluation (deterministic policy)
After training, we run the policy choosing the best action (no randomness), still respecting action masks. We report success rates (≥256/512/1024), p95 max tile, etc.
10) “Expert assist” specifics
A lightweight wrapper around the env. On a small fraction of steps (e.g., 1–5%), it computes an expectimax action and uses that instead of the policy’s pick.
Purpose: inject stronger trajectories occasionally so the policy sees and learns from better sequences it wouldn’t discover quickly on its own.

Need clarification on this section:
"
Each “update cycle”:
Collect a rollout buffer of size n_steps * num_envs by stepping the envs.
Split into mini-batches of size batch_size and run n_epochs passes of SGD.
This repeats until total_steps environment steps are consumed.
Rough counts:
updates ≈ total_steps / (n_steps * num_envs)
gradient steps per update = n_epochs * (n_steps * num_envs / batch_size)
"
Maybe it would be helpful for you to explain it in an example. 
Like for example, the simplest way to do training would be to run it completely in series, one game at a time. Side note I think "game" and "episode" are synonymous in our context?
So one game at a time, would be like you start a game of 2048, and you make up/down/left/right moves until the board fills up and there are no legal moves left, or you hit the episode-steps limit e.g. 6000. Or if we are using curriculum learning like we currently are, that episode could also end once a target tile is reached e.g. 128. 
THEN you would do it all over again, until the total number of up/down/left/right moves (I think these are what we are calling steps) accumulates to "total-steps" e.g. 300M.
But this is slow so we somehow break it up to run games in parallel. What I don't understand is exactly how we break it up. I am generally unfamiliar with the terms "rollout buffer", "epoch", "batch", "SGD", "environment" in this context.


Next questions related to the flags:
why do we choose 64 environments? how did we determine that that is the best number?
I think I really need an ELI5 primer on the rollout and optimization e.g. n-steps, batch size, epochs.
Really you should assume I know next to nothing about machine learning. This is my first project.

In reading the long list of flags, it seems to me that the original script wasn't achieving the goal of learning how to play 2048 so then we just added more features that have made the whole process a lot more complicated. Should we consider maybe taking a step back and stripping out a lot of these features to start from a simpler baseline?