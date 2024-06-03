# EDA and predictive models in particle physics

## Prerequisites and context
This repo is intended to be educational in both exploratory data analysis and predictive models. It uses python and requires some familiarity with that language (or another language typically used for data analysis) and assumes some knowledge of data analysis and machine learning methods. It also requires a good base of mathematics and (to a lesser extent) statistics. Explanations of concepts and potential solutions should be readily available online or in the specific repos referenced below in the acknowledgements.

Install python 3.8 or later and then install the packages in requirements.txt.

## Dimuon_SingleMu_noM dataset
The Dimuon_SingleMu_noM.csv file has had its 'M' (invariant mass) column removed.
Part of the exercises below is to recalculate that value.

Variables in this dataset:
- **Run**: run number of the event
- **Event**: event number - an event is a collision of particles in the detector
- **type#**: type of muon - either G(lobal) or T(racker)
- **Q#**: charge of the muon - +1 or -1 and sum is 0 for each row
- **pt#**: transverse (perpendicular to the beam) momentum of the muon
- **eta#**: pseudorapidity of the muon - a measure of the deflection angle from the particle beam where smaller values indicate greater deflection
- **phi#**: the azimuthal angle (or horizontal/left-right deviation) from the beam
- **E#**: energy of the muon
- **px#,py#,pz#**: x, y, z components of the momentum of the muon

## Exercise 1.1
Initial analysis and visualisation of each of the variables beforehand to get familiar with what each one is, its range of values, some basic stats and histograms, whether any pairs of variables are strongly correlated.

### Questions
1. Are the distributions all relatively smooth or are there obvious aberrations?
2. Which pairs of variables are correlated and why might that be?

## Exercise 1.2
Calculate the invariant mass in two ways for each collision/row in the dataset.
Use each of the equations below to do this. Plot a scatterplot of the invariant mass as estimated by each.

### Invariant Mass equations  

#### Eq. 1
$$M = \sqrt{(E_1+E_2)^2-((p_{1_x}+p_{2_x})^2+(p_{1_y}+p_{2_y})^2+(p_{1_z}+p_{2_z})^2)}$$  

#### Eq. 2
$$M = \sqrt{2p_{T1}p_{T2}(\cosh(\eta_1-\eta_2)-\cos(\phi_1-\phi_2))}$$  

### Questions
1. Which equation will give the more accurate estimate of the invariant mass and why?
2. From the scatterplot, where do you see the greatest deviations between the two estimates?

## Exercise 1.3
Plot a histogram of the invariant mass values (using the more accurate equation) and try to pick out the most obvious peaks.

### Questions
1. What are the values of those peaks?
2. Which kinds of particles might those peaks correspond to?


## pid-5M dataset
The pid-5M.csv data was generated by a GEANT simulation.

Variables in this dataset:
- **id**: -11 (positron), 211 (pion), 2212 (proton), 321 (kaon)
- **p**: momentum (GeV/c)
- **theta**: azimuthal angle (radians)
- **beta**: polar angle (radians)
- **nphe**: number of photoelectrons produced (count)
- **ein**: energy in-bound (GeV)
- **eout**: energy out-bound (GeV)


## Exercise 2.1
Initial analysis and visualisation of each of the variables beforehand to get familiar with what each one is, its range of values, some basic stats and histograms, whether any pairs of variables are strongly correlated.

### Questions
1. Are the distributions all relatively smooth or are there obvious aberrations?
2. Which pairs of variables are correlated and why might that be?

## Exercise 2.2
Stratified split into training and test sets (80:20).
This dataset has 5m rows so the training set has 4m and the test set has 1m at this point. We'll reduce the training set down to 10% of its size to allow us to iterate a bit more quickly with our predictive models.

### Questions
1. Verify that our training data is now 400k rows and that the test set still 1m rows.
2. Check that the proportions of each particle are similar between the two datasets.

## Exercise 2.3
Let's first try a multinomial logistic regression model.
Then we'll try a RandomForestClassifier.

### Questions
1. How does the performance compare between these three models?

## Exercise 2.4
Now we'll try some tweaks...

The most obvious issue with our dataset is that it is imbalanced, which means it will predict the more prevalent classes more often. We don't want this, we want it to predict the correct class independent of its frequency of observation and based only on the input features for a given sample. 

We address this by undersampling highly prevalent classes or oversampling classes with low prevalence or both. We will do this for the training set only because we still want our tests to be representative of the real world, it's fine for imbalances to remain there because a) that will give us a better estimate of performance and b) models don't change as a result of predictions, only training.

Start by undersampling from all of our classes so they are all the same size as the smallest one and rerun the model fitting.

Gradually increase the sample size evenly and refit the model each time to see the effect.

### Questions
1. Considering one class is highly prevalent, we could create a model that always predicts that class and it would be pretty good! What is the accuracy of that baseline model?
2. What is the minimum sample size we need to have a model that is better than the baseline model in 1?
3. As we increase the sample size, at what point do those increases no longer make a difference greater than about 1% to the overall accuracy?

*Note: When we resample from the smaller classes, we are effectively duplicating training data, so we need to be careful not to push this too far because models become less transferrable to new cases when they see the same samples over and over. Rebalancing a dataset is a trade-off between size and diversity.*

## Exercise 2.5
One more tweak we are going to try is rescaling all of our variables. Models tend to fit more slowly (and less well) when all of our input variables are on different scales. By rescaling them all to have mean of zero and a standard deviation of 1, we are putting them all on the same scale but retaining the variation/differences across the samples, albeit in a compressed or expanded form. 

There are a few more things we can try like this that may help - these involve mathematical operations such as logs and power transformations prior to rescaling in order to correct skew.

### Questions
1. What effect does doing this have on the model accuracy?

## Exercise 2.6
Let's build a small neural net with an input layer of 6 neurons, a hidden dense layer of 16 neurons and an output layer of 4 neurons.

### Questions
1. How does this perform compared to the rest?
2. What about when we increase the number of hidden neurons?

## Exercise 2.7
Finally, we're going to gradually adjust the amount of data we use in training to find the sweet spot for training this model.

### Questions
1. What percentage of the original training data do we need to get a model that is right 80% of the time on our test set?
2. What were the most influential things we did above - i.e. what had the greatest effect on model performance?


## Acknowledgements and sources
Many thanks to the following git repos:
https://github.com/jgalser/deeplearning_particlephysics
https://github.com/opendata-education/en_Physics

And these sources for the two datasets:
https://www.kaggle.com/datasets/naharrison/particle-identification-from-detector-responses (pid-5M)
https://opendata.cern.ch/record/545 (DiMuon_SingleMu)
