# EE228 Final Assignment
**Author:** Ruiwen Zhou  
**Student ID:** 518021911150  
**E-mail:** zrw2000917@163.com

## Brief Introduction
This is the final assignment of EE228 Curriculum @ SJTU, in which we are required to generate a learning-based agent to automatically play 2048 game for a high average score in 10 games.  

In this project, I use a completely offline method to obtain a high-performance agent, which has a chance of about 27% to win a single game and more than 70% to reach 1024. It feels awful that I have no luck to produce a log file with 10 victories.  

The main **HIGHLIGHTS** of this project can be summarized as following:
* The training is completely offline, which makes the solution simple.
* The convolution structure of my network is a novel combination of two CNNs.
* The performance of my agent is excellent and stable.

## Code structure
* [`game2048/`](game2048/): the API and Expectimax package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`MyAgent.py`](MyAgent.py): the `MyOwnAgent` class with its implementation.
* [`cnn.py`](cnn.py): the `Conv_Net_Com` class used as agent and its two originals.
* [`train_model.py`](train_model.py): the whole process for training a `Conv_Net_Com` instance, with testing function insides.
* [`data_process.py`](data_process.py): the `Dataset_2048` class to read and store data, based on `torch.utils.data.Dataset`.
* [`generate_data.py`](generate_data.py): utilize the planning-based [`Expectimax`](game2048/expectimax) to generate ideal trajectories as training set.
* [`generate_fingerprint.py`](generate_fingerprint.py): pre-offered script for solution validation.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate self-defined agent.
* [`mdl_final.pkl`](mdl_final.pkl): the stored final version model.
* [`run.sh`](run.sh): the shell script for batch operation on log files.
* [`data_analysis.cpp`](data_analysis.cpp): C++ code, traverse all logs to analyse the performance of agent

## Requirements
* Code tested on Windows and Linux system (Windows 10 and Ubuntu 18.04)
* Code written and tested with PyTorch 1.5.0 and Torchvision 0.6.1
* High version of PyTorch and Torchvision is recommended
* Python 3 (Anaconda 3.6.3 specifically) with numpy, pandas, tqdm and flask

## Usage of CNN Agent
### In Ubuntu terminal
* To train a new CNN agent:
```shell
python train_model.py
# Get training information here:
# ......
Save Model? [Y/n] # press Y to store and N to discard the current model
```
* To evaluate the trained agent:
```shell
python evaluate.py >> EE228_evaluation.log
```
* To generate more logs in a single line, either for higher score or for data aggregation:
```shell
bash run.sh
# Get the current round and average score of each log here
# Example:
# Round: 1/1000, Average score: @10 times 2048.0
# ......
# At last a C++ program will analyze data automatically
# Some insights about the performance of agent will be listed here
# Example:
# Average score per game: 2048.0
# ......
```
With parameters set in [`run.sh`](run.sh), certain number of log files can be obtained and then analyzed easily.

### In Windows Powershell
* To train and evaluate an agent, the operation is exactly the same as in Ubuntu terminal.
* `run.sh` cannot be used in Windows due to the lack of GPU support in WSL.  
(Things might change by the end of this year, as Microsoft plans to publish an update for this)

## Performance analysis
Using `mdl_final.pkl` in agent, I generated 15,000 logs in total and provide some results here:
```text
Average score per game:
Max average score @10 times ever obtained:
Ratio of AVERAGE score > 1024 @10 times:
Ratio of SINGLE GAME victory:
```
### Max tile statistics:
Max Tile | 2~32 | 64 | 128 | 256 | 512 | 1024 | 2048
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
Frequence | 0 | 1 |
Frequency | 0% |

## API Tutorial
For tutorial of the given 2048 API, refer to [API Tutorial](https://github.com/duducheng/2048-api/blob/master/README.md).

## LICENSE
The code is under Apache-2.0 License.
