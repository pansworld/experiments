# Experimnts in Machine Learning

## Elevator Experiments
### Premise
AI is designed to be statistically accurate bot not be human. It needs to think like a human to pass the turing test. This is a basic experiment where the emphasis is on being as accurate as a Human rather than being accurate overall.

### Objective
I have 3 elevators in my building. The objective is to figure out how can a ML program Predict what I am predicting. What are the benefits of that model?

### Process
#### Data Collection
* For each trip note down which elevator comes (up or down does not matter)
* Sum up the number of times an elevator comes on a particular day

### Prediction Process

#### Alogrithm 1
* From total running counts calculate the overall probability for each elevator
* The probability based on the day of the week
* Predict using Reinformcement Learning using this Rewards Function


| Rewards        | Human Prediction vs Machine Prediction | Human Prediction vs Actual | Machine Prediction vs Actual |
|----------------|----------------------------------------|----------------------------|------------------------------|
| Matches        | 1                                      | 0.5                        | 0.5                          |
| Does not Match | -1                                     | -0.5                       | -0.5                         |

* Find the policy that matches up with Human accuracy

