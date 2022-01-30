# Non-preemptive dynamic mixed criticality scheduling problem using Reinforcement learning
----------------
## File Hierarchy

#### environment-generator.py
- The core of the agent, has all functions regarding building custom environments
- All reward, punishment and agent interactions are developed here.

#### filter.py
- Responsible for filtration and quantization of the time horizon using non-preemptive EDF

#### main.py
- Training agent starting point is here.
- Has the DQN Model functions

#### online-testing-evaluation.py
- Updated evaluation framework for pre-generated datasets for online environment
- It is configured to evaluate on the previously generated datasets from DatasetGeneration.py file
- Initially loads the trained agent from ray_results folder, then loads checkpoint and starts evaluation
- Results are recoreded into textfiles and graphs.

#### offline-evaluation.py
- Original file for evaluation of the agent
- Evaluation is done using data generated on the fly, not pre-generated sets.

#### job-generator.py
- Generate each job instance based on the work by [Baruah and Guo(2013)](https://ieeexplore.ieee.org/document/6728862)

#### DatasetGeneration.py

- Generate dataset of job instances based on specific parameters:
-- Number of Jobs, job generation characteristics and number of instances.

#### TrainingDataAnalysis.py

- Load the training data which was generated on the fly and analyses the data to find average characteristics of training dataset