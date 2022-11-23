## U.S. Soccer Federation 2023 MIT Sloan Sports Analytics Conference Research Repository. 

This repository is provided alongside our paper: _"A Graph Neural Network deep-dive into successful counterattacks"_.
It contains an interactive [Python Jupyter Notebook](counterattack.ipynb) for training GNNs using the [Spektral](https://graphneural.network/) library to try and improve upon our research.


### Research Abstract
The purpose of this research is to build gender-specific, first-of-their-kind Graph Neural Networks to model the likelihood of a counterattack being successful
and uncover what factors make them successful in both men's and women's professional soccer. 
These models are trained on a total of 20,863 frames of algorithmically identified counterattacking sequences from synchronized StatsPerform on-ball 
event and SkillCorner spatiotemporal (broadcast) tracking data. This dataset is derived from 632 games of MLS (2022), NWSL (2022) and international women’s soccer (2020-2022).
This data, linked to at the bottom of this page, is automatically loaded in the [Counterattack Jupyter Notebook](counterattack.ipynb).

With this data we demonstrate that gender-specific Graph Neural Networks outperform architecturally identical gender-ambiguous models in predicting the successful outcome of 
counterattacks. We show, using Permutation Feature Importance, that byline to byline speed, angle to the goal, angle to the ball and sideline to sideline speed are the node
features with the highest impact on model performance.

### Installing Jupyter Notebook using pip:

  To install Jupyter using pip, first check if pip is updated in your system. Use the following command in the command prompt to update pip
    
  ```
  python -m pip install --upgrade pip
  ```

  Once upgraded, use the following command to install Jupyter
  ```
  python -m pip install jupyter
  ```
  
  Use the following command to launch Jupyter Notebook
  ```
  jupyter notebook
  ```
  
  Once launched, you should be able to see the following screen
  
  <img width="1188" alt="image" src="https://user-images.githubusercontent.com/108815194/203169206-3526ced4-e63d-44b5-99bc-2d0075238742.png">


### Installing the required Python libraries:
  Make sure the requirement.txt file is in the directory. Navigate to the directory using command prompt and type the following command
  ```
  pip install -r requirements.txt
  ```

### Running the script
  * Navigate to the location where you have cloned the GitHub repository and open the interactive notebook.
  * Open the .ipynb file.
  * Run the first cell by clicking the play button or using the shortcut ```Shift + Enter``` through your keyboard. If all the libraries are installed succesffuly this code block should execute without throwing any errors.
  * Run the subsequent cell blocks to load the data.
  * Choose a file between men's data, women's data and combined dataset using the dropdown list.
    * Men's dataset - MLS 2022 season
    * Women's dataset - NWSL 2022 season + International Women's soccer
    * Combined dataset - Combination of both men and women's data
  * Choose the adjacency matrix from the dropdown list.
    * normal - connects attacking players amongst themselves, defensive players amongst themselves and the attacking and defending players are conencted through the ball.
    * delaunay - connects a few attacking players and defending players in a delaunay matrix fashion
    * dense - connects all the players and the ball to each other
    * dense_ap - connects all the attacking players to each other and defensive players.
    * dense_dp - connects all the defending players to each other and attacking players.
  
  * Choose the Edge features and Node Features using the checkboxes.
    * Edge Feature options:
      * Player Distance - Distance between two players connected to each other
      * Speed Difference - Speed difference between two players connected to each other
      * Positional Sine angle - Sine of the angle created between two players in the edge
      * Positional Cosine angle - Cosine of the angle created between two players in the edge
      * Velocity Sine angle - Sine of the angle created between the velocity vectors of two players in the edge
      * Velocity Cosine angle - Coine of the angle created between the velocity vectors of two players in the edge
      
    * Node Feature options:
      * x coordinate - x coordinate on the 2D pitch for the player / ball
      * y coordinate - y coordinate on the 2D pitch for the player / ball
      * vx - Velocity vector's x coordinate
      * vy - Velocity vector's y coordinate
      * Velocity - magnitude of the velocity
      *  Velocity Angle - angle made by the velocity vector
      * Distance to Goal - distance of the player from the goal post
      * Angle with Goal - angle made by the player with the goal
      * Distance to Ball - distance from the ball (always 0 for the ball)
      * Angle with Ball - angle made with the ball (always 0 for the ball)
      * Attacking Team Flag - 1 if the team is attacking, 0 if not and for the ball
      * Potential Receiver - 1 if player is a potential receiver, 0 otherwise
    
  * Update the graph neural training network configurations as per your requirement. You may also try chaning the network layers.
  * Start the model training. It should stop after the epochs are completed. Check your training logloss score. If it is not satisfactory rerun the training block.
  * Use the block for testing model logloss and ROC-AUC curve. 
  * Further, you may opt to look at model calibration and also calculate the Expected Calibration Error (More details in the notebook text blocks).
  * It is also possible to check which features contribute the most to the model performance. (**Note**: The ```Attacking Team Flag``` checkbox from the Node Features needs to be selected to calculate the feature importance.) Choose between attacking and defending players and note the differences via the box plot.

------

### Citation

If you use any of the data or files within this repository, please cite our paper.

------

### Data

The data loading process is automated within the [Counterattack Jupyter Notebook](counterattack.ipynb), but it can also be obtained through the links below.

- [Counterattacks Women](https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/women.pkl)
- [Counterattacks Men](https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/men.pkl)
- [Counterattacks Combined](https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/combined.pkl)

-----

### Requirements

- Python 3.9+
