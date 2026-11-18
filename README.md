# Evaluating Rate Limiting and Lightweight Machine Learning Under Bursty HTTP Traffic: A Simulation Study

## About This Project

I started this project because I wanted to understand what happens when a web server receives too many HTTP requests, especially when some of those requests come from legitimate users and others come from attackers.

There are different ways to deal with this kind of traffic, but I was curious about how they compare under the same conditions. For this reason, I built a simulation in Python and tested fixed rate limiting, adaptive rate limiting, and a machine learning-based approach. I also ran the experiments without any defense to have something to compare the results against.

One thing I wanted to pay attention to was the experience of legitimate users. A defense might block a large number of attack requests, but that does not necessarily mean it is doing a good job if normal users cannot access the server either.

## How the Simulation Works

I used SimPy to simulate a server receiving HTTP requests from different types of clients.

I created four traffic scenarios:

- **Normal traffic:** Regular users sending requests at a relatively low rate.
- **Flash crowd:** A sudden increase in legitimate requests.
- **Attack traffic:** Normal users and attackers sending requests at the same time.
- **Combined traffic:** A flash crowd happening at the same time as an attack.

For each scenario, I compared four configurations: no defense, fixed rate limiting, adaptive rate limiting, and machine learning-based filtering.

The machine learning model is a Random Forest classifier trained on synthetic traffic data. I also experimented with different numbers of inference workers to see what happens when the classifier has limited processing capacity.

This was an important part of the experiment for me because machine learning takes time to process requests. I wanted to see whether that extra processing could become a problem when the server is already receiving a lot of traffic.

## What I Measured

I mainly looked at:

- How many legitimate requests were completed
- How many attack requests were rejected
- How long legitimate requests took to complete
- How many requests were dropped when queues became full
- How the number of machine learning inference workers affected the results

I saved the experiment results as CSV files so I could compare the different scenarios and defense methods.

## Tools I Used

The simulation was written in Python.

I used **SimPy** for the discrete-event simulation, **scikit-learn** for the Random Forest model, and **joblib** to save and load the trained model.

The required Python packages are listed in `requirements.txt`.

## Running the Project

First, download or clone the repository and open its folder in a terminal.

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

The repository already includes the trained Random Forest model, so you do not need to train it again just to run the simulation.

If you want to retrain the model, you can use:

```bash
python train_ml_model.py
```

To run the main simulation with the smaller set of experiments:

```bash
python experiment_v14.py
```

To run the full experiment:

```bash
python experiment_v14.py --full
```

The scripts save their CSV output in the main project folder. I have also included copies of the experiment results in the `results` folder.

The full experiment takes longer to run, so I recommend starting with the smaller run if you just want to see how the simulation works.

## A Few Things to Keep in Mind

This project does not send attacks to a real server. Everything takes place in a simulated environment, and the HTTP traffic is generated synthetically.

The results depend on the assumptions I made about request rates, server capacity, queue limits, and defense settings. They are useful for comparing the methods within this simulation, but they should not be treated as proof that the same results would happen on a real network.

I also do not consider the machine learning model a perfect solution. Part of the reason I tested different inference capacities was to understand its limitations, not just its ability to identify attack traffic.

## Author

Nesrin Kara  
MSc Student in Computer Engineering  
European University of Lefke