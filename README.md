## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). 
* It has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

### Dataset metadata
|Variable|Meaning|Values|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorchSF|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements
The client has received an inheritance from a deceased great-grandfather located in Ames, Iowa. They aim to maximise the sale price for each of their inherited properties.

The client has an excellent understanding of property prices in their own state and residential area, however this knowledge may not generalise, and so might lead to inaccurate appraisals; consequently they seek the help of a data practitioner to assist in accurately valuing their properties. To this end, the client desires to know what makes a house desirable and valuable in Ames, Iowa.

The client has located a public dataset with house prices for Ames, Iowa.

To summarise in precise terms:

* 1 - The client is interested in discovering how the house attributes correlate with the sale price in Ames, Iowa. Therefore, the client expects data visualisations of the correlated variables against the sale price to illustrate any relationships.
* 2 - The client is interested in predicting the house sale price for their four inherited houses, and more generally any other house in Ames, Iowa.


## Hypothesis and how to validate?
### Hypothesis 1:
The features are significantly inter-correlated. More specifically some of the features are groupable into sets, based on a strong correlation amongst members, and where each set may also be less strongly correlated to other sets. A set of closely correlated features may well be replaceable by a single representative feature,
with regard to its correlation with sale price. The respective groupings expected are as follows:

Feature Group 1 (Size group):
Expect a strong positive correlation between features.
* First Floor square feet (1stFlrSF)
* Second Floor square feet (2ndFlrSF)
* Bedrooms above grade (BedroomAbvGr)
* Total square feet of basement area (TotalBsmtSF)
* Above grade (ground) living area square feet (GrLivArea)
* Type 1 finished square feet (BsmtFinSF1)
* Unfinished square feet of basement area (BsmtUnfSF)


Feature Group 2 (Quality group): 
Expect a moderate to strong positive correlation between features.
* YearBuilt: original construction date
* YearRemodAdd: Remodel date
* OverallCond: Rates the overall condition of the house
* OverallQual: Rates the overall material and finish of the house
* KitchenQual: Kitchen quality

Feature Group 3 (Lot group): Expect a moderate positive correlation
* LotArea: Lot size in square feet
* LotFrontage: Linear feet of street connected to property

Feature Group 4 (Garage group): Expect a moderate positive correlation
* GarageFinish: Interior finish of the garage
* GarageYrBlt: Year garage was built

The remaining ungrouped features are expected to express a correlation with other features to a lesser extent.

### Hypothesis 2:
The size group features, are the most significant for predicting the sale price of a property.

### Hypothesis 3:
The quality group features, are the next most significant in predicting the sale price of a property.

### Hypothesis 4:
The remaining features are expected to correlate with the target sale price, ordered by strength, as displayed in the table below:

|Feature(s)|Correlation strength|Correlation sign|
|:----|:----|:----|
|Group 3|moderate|positive|
|BsmtFinType1|moderate|positive|
|GarageArea|moderate|positive|
|Group 4|moderate|positive|
|EnclosedPorchSF|weak|positive|
|BsmtExposure|weak|positive|
|WoodDeckSF|weak|positive|
|OpenPorchSF|weak|positive|
|MasVnrArea|weak|positive|

### Validation Methods:
In general all hypotheses will be validated using correlation tests, and other statistical tests to test whether the relevant features and target have statistically significant relationships. Plots will be used where appropriate to help visualise and also verify any relationships indicated by the tests. 

Hypothesis 1:
The truth of this hypothesis will be evaluated using the correlation type test results, by assessing whether all feature group members demonstrate the expected
correlation, and or whether other features should be included in a group.

Hypothesis 2,3 & 4:
All features will be ranked with regard to their Predictive Power Score(PPS) and correlation coefficient with the target: sale price. This will reveal the validity of the hypotheses.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

**Business requirement 1**: discover how house attributes correlate with the 
sale price of houses in Ames, Iowa.

### BR1 User stories
* **As the client, I want** to be able to view information indicating the relationships between house attributes, **so that I can** better understand which attributes are related, thus simplifying the number of attributes that need to be considered independently.
* **As the client, I want** to be able to see how the house attributes relate to the sale price, **so that I can** understand which attributes are most significant for determining the sale price.
* **As the client, I want** to be able to view visual plots, **in order to** better perceive the relationships between house attributes including the sale price.
* **As the client, I want** to be able to access all summary and in-depth annotated information and plots on a dashboard, **so that** it is easily retrievable.
* **As a data practioner**, I want to be able to view on the dashboard the project hypotheses, as well as how they were validated, **so that I can** come to my own conclusions.

### BR1 User story tasks
* Collect housing data from kaggle.
* Clean data, if necessary, so that it is fit for analysis.
* Perform initial exploratory data analysis on each feature and the target individually; in order to inform the choice of correlation and other statistical tests to perform. Also needed for data understanding in the ML tasks related to BR2.
* Generate a set of correlation matrices with components for all feature-feature and feature-target pairs.
* Generate a Predictive Power Score(PPS) matrix with components for all feature-feature and feature-target pairs.
* Produce a series of plots for all pairs to illustrate any relationships.
* Create project summary dashboard page.
* Create dashboard page displaying the results of the feature-feature and feature-target pairs analysis, with the use of plots.

**Business requrement 2**: predict sale price of houses in Ames, Iowa.

### BR2 User stories
* **As the client, I want** to be able view on a dashboard the predicted sale price of my four properties, **so that I can** maximise the achievable sale price for each of them.
* **As the client, I want** to be able to enter on the dashboard the house attribute values for any property in Ames, Iowa and see the predicted price, **so that I can** see how changing attributes affects the sale price; it also would allow me to update the sale price if I modify one of my properties before selling.
* **As a data practioner, I want** to be able to evaluate the model performance,
**so that I can**, understand what works well, and how to improve the model.

### BR2 User story tasks
* Perform exploratory data analysis for the purpose of data understanding as part of the CRISP-DM workflow.
* Perform housing data cleaning/preparation.
* Perform feature engineering/scaling.
* Train model for predicting sale price from house attributes.
* Optimise the model and perform hyperparameter tuning.
* Evaluate the model performace.
* Finalise the model pipeline once the success criteria are met.
* Create a dashboard page displaying the attributes and sale prices of the client's four properties.
* Create a dashboard page/section allowing the client to enter a set of house attributes, and calculate the predicted sale price dynamically.
* Create a dashboard page displaying the final model performance. 

## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* In case you would like to thank the people that provided support through this project.

