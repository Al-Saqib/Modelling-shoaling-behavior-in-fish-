# Impact of Variable Shoaling Size and Predator Presence on the Individual Interactions and Collective Motion of Shoaling Fish

## Description

This project investigates the impact of variable shoaling size and predator presence on the individual interactions and collective motion of shoaling fish. Using agent-based modelling, the study explores how different interaction rules affect shoaling behaviour. Three models are implemented: the main model with basic interaction rules, an extended model with alignment behaviour, and a conventional model considering multiple neighbours. The study aims to understand how these factors influence the stability of shoaling behaviour and survival chances in the presence of a predator.

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Al-Saqib/Modelling-shoaling-behaviour-in-fish.git
```

2. Navigate to the project directory:

```bash
cd your-repo-name
```

3. Extract the contents of `Relevant Codes and Figures.zip` to the project directory.

4. Run the Python scripts for the different models:

- For the main model:
  ```bash
  python "Implemented Codes/main/main_model.py"
  ```

- For the extended model:
  ```bash
  python "Implemented Codes/extended/extended_model.py"
  ```

- For the conventional model:
  ```bash
  python "Implemented Codes/conventional/conventional_model.py"
  ```

- To generate plots for the results:
  ```bash
  python "Implemented Codes/main/main_model_plots.py"
  python "Implemented Codes/extended/extended_model_plots.py"
  python "Implemented Codes/conventional/conventional_model_plots.py"
  ```

- To include predator interactions, run:
  ```bash
  python "Implemented Codes/main/main_model_predator.py"
  python "Implemented Codes/extended/extended_model_predator.py"
  python "Implemented Codes/conventional/conventional_model_predator.py"
  ```

## Project Structure

- `Implemented Codes/main/`: Contains the main model and associated scripts.
- `Implemented Codes/extended/`: Contains the extended model with alignment behaviour.
- `Implemented Codes/conventional/`: Contains the conventional model considering multiple neighbours.
- `Figures/`: Contains figures for the results.
- `Code Figures/`: Contains code snippets used in the report.
- `3d fish.py`: Generates a three-dimensional animation plot for the main model.

## Features

### Simulation Scenarios

- **Variable Shoaling Size**: Simulates shoaling behaviour with school sizes ranging from 10 to 100.
- **Predator Presence**: Examines the impact of a predator on shoaling behaviour and survival rates.
- **Three Models**:
  - Main Model: Basic attraction and repulsion behaviours.
  - Extended Model: Includes alignment behaviour.
  - Conventional Model: Considers interactions with multiple neighbours.

### Metrics

- **Cohesion**: Measures the ability to maintain position close to the center of the shoal.
- **Separation**: Measures the distance between each fish and its neighbours.
- **Alignment**: Measures how similar the velocity of each fish is to its neighbours.
- **Percentage of Fish Caught**: Evaluates survival rates in the presence of a predator.

## Results

The project compares the outcomes of simulations under different scenarios, highlighting how shoaling size and predator presence affect the shoaling behaviour and survival chances of fish. The results are presented in detailed plots generated by the scripts.

## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request detailing the changes you have made.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or inquiries, please contact me at saqib.majumder01@gmail.com.
