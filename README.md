# Ireland_Housing_Prices
Analysis of Irish Housing Prices

<figure>
    <img src='./media/derelict_house.jpg'>
    <figcaption>Geograph - https://www.geograph.ie/photo/3116632</figcaption>
</figure>

## Introduction
This project will use data from The Residential Property Price Register (RPPR) to create
various Machine Learning models to predict property prices in Ireland, then compare the performances
of the models.<br>

## Requirements
Requirements can be installed using [requirements.txt](https://github.com/21chubaka/Ireland_Housing_Prices/blob/main/requirements.txt).
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Sklearn<br>

## Data
Property Services Regulatory Authority (PSRA) - The Residential Property Price Register (RPPR) for 2010-2022<br>
Features:<br>
- Date of Sale (dd/mm/yyyy):    object
- Address:                      object
- Postal Code:                  object
- County:                       object
- Price (€):                    object
- Not Full Market Price:        object
- VAT Exclusive:                object
- Description of Property:      object
- Property Size Description:    object<br>

Central Statistics Office provided by the government of Ireland - Income Per Person and Income Indices by County by Year for 2010-2022<br>
Features:<br>
- Year:                                  int64
- County:                              string
- Income_Indices:                      float64
- Income_Per_Person_euro:              float64

## Data Cleaning
Intial data exploration was carried out to better understand the RPPR data and identify any data cleaning that needed to be carried out before
modeling.<br>
### Initial Descriptive Statistics on Continuous Features
<figure>
    <img src='/media/descrip_stats_cont_feat.png'>
</figure>

### Initial Descriptive Statistics on Categorical Features
<figure>
    <img src='/media/descrip_stats_cat_feat.png'>
</figure>

Some integrity checks on the dataset were:
- Check for Null Values by Column
- Check for Negative Price Values
- Check for Future Dates
- Check for Month Values Greater than 12 or Less than 1
- Check for Day Values Greater than 31 or Less than 1
- Check for Year Values Older than 2010
- Check for 'Second-hand Dwellings' with 'Yes' VAT Exclusive
- Check for Outliers by Column
- Etc.

### 'Price (€)' feature
The 'Price (€)' feature contained outliers that were signficantly skewing the data, especially upper bound outliers.  The impact of the outliers
can be observed from the boxplot below.
<figure>
    <img src='/media/rppr_price_boxplot.png'>
</figure>

### 'Postal Code' feature
The 'Postal Code' feature had 8,086 rows with NULL values and some with missing values.  This feature also had some Dublin Postal Codes labeled as
'Baile Atha Cliath', which is Dublin in Irish.

### 'Description of Property' feature
The 'Description of Property' feature had some labels in Irish.

### 'Property Size Description' feature
The 'Property Size Description' feature had 8,984 rows with NULL values.  This feature also had two labels for the same size description of
'greater than 125 sq meters'.

### Data Cleaning Plan
<figure>
    <img src='/media/data_cleaning_plan1.png'>
</figure>
The above image layouts out the data issues identified with the RPPR dataset and the handling strategies employed to clean the dataset.
For some of the issues it was decided to leave them as is, due to the small impact or the goal of keeping data intact. For example, 
after researching the lower bound outliers of the property prices, it was found that those prices and properties were normal (non-developer) 
properties that an individual would purchase.<br>
As an Irish Language speaker and supporter it pained me, but for the purposes of this project any Irish labels were translated for the continuity of data.<br>
It should be noted, that in addition to the data cleaning plan, there were some initial changes to feature names, exculsions of symbols in the price feature, and updates to data types to facilitate better data transformations and analysis.

### Income by County Dataset - Central Statistics Office
The final step of data cleaning was combining the RPPR and Income by County datasets.  This was completed by using a left join on 'Year' and 
'County'.

## Exploratory Analysis - RPPR & Income Data
### Target Feature - Price
<figure>
    <img src='/media/price_hist.png'>
</figure>

### Continuous Features
#### Descriptive Statistics on Continuous Features
<figure>
    <img src='/media/final_descrip_stats_cont_feat.png'>
</figure>

#### Correlation Statistics on Continuous Features
<figure>
    <img src='/media/cont_corr.png'>
</figure>

#### Correlation Matrix on Continuous Features
<figure>
    <img src='/media/cont_corr_matrix.png'>
</figure>

Some observations from the matrix:
- 'Income_Indices' & 'Income_Per_Person_euro' positively correlated to price.
- 'Income_Indices' are based off of 'Income_Per_Person_euro' and 'Income_Per_Person_euro' is better correlated, so I will focus on 'Income_Per_Person_euro' to start.

#### Income Per Person vs. Price
<figure>
    <img src='/media/income_price_scat.png'>
</figure>

#### Year vs. Price
<figure>
    <img src='/media/year_price_boxplot.png'>
</figure>

#### Continuous Feature Conclusions
<b>Year</b>:<br>
After doing some analysis, I have decided to include Year as a feature to start. There was a correlation to Price (albeit on the lower end), but it did offer information on how price changed over time (down 2011 and up after 2013). The recession may have negatively affect the correlation number.<br>

<b>Income Per Person</b>:<br>
This is an easy feature to choose to include in my models. It was the stronest correlated feature.
Also as mentioned above, while Income Indices is another well correlated feature I will not include it at this time because it is value that is based off of Income per Person.<br>

### Categorical Features
#### Descriptive Statistics on Categorical Features
<figure>
    <img src='/media/final_descrip_stats_cat_feat.png'>
</figure>

Some observations:
- 'Postal_Code' has a high amount of unknown values due to Dublin being the only county that tends to use postal code in that manner.
- 'County' having 26 makes sense due to the 26 counties in the Republic of Ireland.
- 'Property_Size_Description' has an extremely high amount of unknown values and will most likely not be included in the models at this time.

#### Postal Code vs. Price
<figure>
    <img src='/media/postal_price_boxplot.png'>
</figure>

First, it does show Dublin tends to use postal codes. Also, it does show the price differences with a high level of variation by Dublin postal code.<br>
From the box plot of postal code vs price, I can dig deeper into Dublin alone, and its specified areas. The highest median prices are Dublin 14, 4, 6, 6w, and 16 – all between 400,000 to 600,000 euro. I noted that these are all even numbered postal codes and therefore are South of the River Liffey. Dublin 6 has the widest interquartile range. The lowest median price was Dublin 10, followed closely by Dublin 11 and Dublin 22. The specific area within Dublin does appear to play an important factor in price.

#### County vs. Price
<figure>
    <img src='/media/county_price_boxplot.png'>
</figure>

From the box plot of county vs price, I can easily tell Dublin, the Wicklow, then Kildare, then Meath have the highest median prices – all about 200,000 Euro. Dublin and Wicklow counties have the widest spread and therefore variation as well, both for the interquartile range as well as whiskers. There are several counties where the 75th percentile does not even reach 200,000, such as Carlow, Cavan, Donegal, Laois, Leitrim, Longford, Mayo, Monaghan, Offaly, Roscommon, Sligo, and Tipperary. Leitrim, Longford, and Mayo have the lowest median prices. Leitrim also has the narrowest interquartile range as well as whiskers. It could be useful to explore single counties more, such as Dublin by its postal codes, as well as numerical data specific to each county, such as population size or income levels.<br>
I find this quite informative of the price variations by county, making it a decent feature to include in the models.

#### Categorical Feature Conclusions
<b>Postal Code</b>:<br>
I am going to include Postal Code because it is a strong representive of Dublin, which is the most frequent county in the data. That being said, in future analysis/models I want to seperate the data as a Dublin dataset and then the rest of Ireland dataset.

<b>County</b>:<br>
I am going to include County because it is the most complete feature. The county is captured for every sale. Moreover, County also holds spacial/geographical information accross Ireland.

## Models
For this project, the goal was to compare the performances of predicting house prices of Multiple Linear Regression, Decision Tree, and Random Forest models.<br>
The cleaned dataset of RPPR/Income was used and the Postal Code and County features were categorically encoded, as they showed promised from the exploratory analysis.  Then the dataset was randomly shuffled and split using the 70/30 train/test split.  The shuffled complete dataset will be used for 5-fold Cross-Validation.

### Features
The features used for the models:
- Year
- Income Per Person
- Postal Code (categorically encoded)
- County (categorically encoded)

## Performance

### Metrics
The performance of the models will be reviewed on:
- Training dataset
- Test dataset
- Complete shuffled dataset

#### The Mean Absolute Error (MAE)
The error of each row is computed, then absolute value taken, and finally averaged across all rows.<br>

#### The Mean Squared Error (MSE)
The error of each row is computed, then squared value taken, and finally averaged across all rows.<br>

#### The Root Mean Squared Error (RMSE)
This error is the difference between the actual price and predicted per the model.  This is calculated for each row and then squared.  In the end, the squared error is then averaged, and finally square root is taken.  This is interpreted in the units of the variable being predicted, in this case Property Price.  The lower the RMSE, the better the model.<br>

#### The R-squared Score (R2)
This is typically in the range of 0 to 1 and is calculated by the sum of all squared errors, divided by the total sum of squares (predictions), subtracted from 1.  R-squared is not dependent on original units (property price), and explains how much better the model is over predicting simply the average of the past prices.<br>

#### 5-fold Cross-Validation
Used to gauge the ablity of a model to predict on new data.  The dataset is split into k number of folds (5 in this project) and its performance is tested on each fold.  Then the scores are averaged to achieve its cross-validated score.

### Multiple Linear Regression
<table>
    <tr>
        <th>Metric</th>
        <th>Training</th>
        <th>Test</th>
        <th>5-Fold Cross-Validation</th>
    </tr>
    <tr>
        <td>MAE</td>
        <td>112672.37</td>
        <td>115194.74</td>
        <td>113429.41</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>27799942517.62</td>
        <td>30464257708.54</td>
        <td>28599488564.33</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>166733.15</td>
        <td>174540.13</td>
        <td>169041.30</td>
    </tr>
    <tr>
        <td>R2</td>
        <td>0.2690</td>
        <td>0.2522</td>
        <td>0.2630</td>
    </tr>
</table>

#### Training
Based on the MAE, this model is off by 112,672 euro on average. Considering the majority of properties were less than 200,000 euro this is a large proportion to be off by. The RMSE is even higher due to it punishing higher outliers.<br>
Given those metrics, the low R-squared score of .2690 makes sense. There is ample room to improve!<br>

#### Test
There is a marginal decrease in R-squared (.2522) and an increase in RMSE (174,540 euro), while running it on the test data.<br>

#### 5-Fold Cross-Validation
The evaluation metrics were quite similiar accross Train, Test, and Cross. It was expected that Linear Regression would not perform well, given that this is not a linear issue.<br>

### Decision Tree
In short, Decision Tree is an approach of supervised machine learning that makes predictions by using a tree of questions whose answers inform its prediction.<br>

For this project I used Decision Tree Regressor also known as a Continuous Variable Decision Tree. I choose this approach due to two factors. First, the target feature I am trying to predict is not a binary outcome; it is a numerical value of price of a property. Secondly, I wished to use multiple features in the model, which a Continuous Variable Decision Tree is used for.
Source: https://www.mastersindatascience.org/learning/introduction-to-machine-learning-algorithms/decision-tree/#:~:text=A%20decision%20tree%20is%20a,that%20contains%20the%20desired%20categorization.

Below I have included both the textual and graphical respresentation of the Decision Tree Regressor tree for my model:<br>
|--- feature_1 <= 30842.25
|   |--- feature_1 <= 26827.85
|   |   |--- feature_1 <= 24687.81
|   |   |   |--- feature_0 <= 2010.50
|   |   |   |   |--- feature_1 <= 20295.07
|   |   |   |   |   |--- value: [321420.00]
|   |   |   |   |--- feature_1 >  20295.07
|   |   |   |   |   |--- feature_3 <= 1.50
|   |   |   |   |   |   |--- feature_1 <= 23403.89
|   |   |   |   |   |   |   |--- value: [130000.00]
|   |   |   |   |   |   |--- feature_1 >  23403.89
|   |   |   |   |   |   |   |--- value: [133203.12]
|   |   |   |   |   |--- feature_3 >  1.50
|   |   |   |   |   |   |--- feature_3 <= 9.50
|   |   |   |   |   |   |   |--- feature_3 <= 4.50
|   |   |   |   |   |   |   |   |--- value: [250386.60]
|   |   |   |   |   |   |   |--- feature_3 >  4.50
|   |   |   |   |   |   |   |   |--- value: [221781.70]
|   |   |   |   |   |   |--- feature_3 >  9.50
|   |   |   |   |   |   |   |--- feature_3 <= 14.50
|   |   |   |   |   |   |   |   |--- value: [141994.85]
|   |   |   |   |   |   |   |--- feature_3 >  14.50
|   |   |   |   |   |   |   |   |--- value: [194006.45]
|   |   |   |--- feature_0 >  2010.50
|   |   |   |   |--- feature_0 <= 2017.50
|   |   |   |   |   |--- feature_0 <= 2011.50
|   |   |   |   |   |   |--- feature_1 <= 23204.71
|   |   |   |   |   |   |   |--- feature_1 <= 21757.81
|   |   |   |   |   |   |   |   |--- value: [134117.74]
|   |   |   |   |   |   |   |--- feature_1 >  21757.81
|   |   |   |   |   |   |   |   |--- value: [158638.58]
|   |   |   |   |   |   |--- feature_1 >  23204.71
|   |   |   |   |   |   |   |--- feature_1 <= 23862.66
|   |   |   |   |   |   |   |   |--- value: [260000.00]
|   |   |   |   |   |   |   |--- feature_1 >  23862.66
|   |   |   |   |   |   |   |   |--- value: [200678.16]
|   |   |   |   |   |--- feature_0 >  2011.50
|   |   |   |   |   |   |--- feature_1 <= 23326.65
|   |   |   |   |   |   |   |--- feature_1 <= 18907.33
|   |   |   |   |   |   |   |   |--- value: [57768.89]
|   |   |   |   |   |   |   |--- feature_1 >  18907.33
|   |   |   |   |   |   |   |   |--- value: [106296.44]
|   |   |   |   |   |   |--- feature_1 >  23326.65
|   |   |   |   |   |   |   |--- feature_3 <= 6.50
|   |   |   |   |   |   |   |   |--- value: [144863.21]
|   |   |   |   |   |   |   |--- feature_3 >  6.50
|   |   |   |   |   |   |   |   |--- value: [126213.70]
|   |   |   |   |--- feature_0 >  2017.50
|   |   |   |   |   |--- feature_1 <= 22993.58
|   |   |   |   |   |   |--- feature_3 <= 15.50
|   |   |   |   |   |   |   |--- feature_1 <= 21667.29
|   |   |   |   |   |   |   |   |--- value: [112406.25]
|   |   |   |   |   |   |   |--- feature_1 >  21667.29
|   |   |   |   |   |   |   |   |--- value: [104856.76]
|   |   |   |   |   |   |--- feature_3 >  15.50
|   |   |   |   |   |   |   |--- feature_1 <= 22412.43
|   |   |   |   |   |   |   |   |--- value: [189000.00]
|   |   |   |   |   |   |   |--- feature_1 >  22412.43
|   |   |   |   |   |   |   |   |--- value: [123076.59]
|   |   |   |   |   |--- feature_1 >  22993.58
|   |   |   |   |   |   |--- feature_3 <= 2.50
|   |   |   |   |   |   |   |--- value: [113068.72]
|   |   |   |   |   |   |--- feature_3 >  2.50
|   |   |   |   |   |   |   |--- feature_1 <= 23136.18
|   |   |   |   |   |   |   |   |--- value: [201822.93]
|   |   |   |   |   |   |   |--- feature_1 >  23136.18
|   |   |   |   |   |   |   |   |--- value: [168121.20]
|   |   |--- feature_1 >  24687.81
|   |   |   |--- feature_3 <= 24.50
|   |   |   |   |--- feature_0 <= 2011.50
|   |   |   |   |   |--- feature_1 <= 25528.25
|   |   |   |   |   |   |--- feature_1 <= 25049.13
|   |   |   |   |   |   |   |--- value: [195964.44]
|   |   |   |   |   |   |--- feature_1 >  25049.13
|   |   |   |   |   |   |   |--- feature_3 <= 9.00
|   |   |   |   |   |   |   |   |--- value: [130510.00]
|   |   |   |   |   |   |   |--- feature_3 >  9.00
|   |   |   |   |   |   |   |   |--- value: [187633.33]
|   |   |   |   |   |--- feature_1 >  25528.25
|   |   |   |   |   |   |--- feature_0 <= 2010.50
|   |   |   |   |   |   |   |--- feature_3 <= 5.50
|   |   |   |   |   |   |   |   |--- value: [245557.90]
|   |   |   |   |   |   |   |--- feature_3 >  5.50
|   |   |   |   |   |   |   |   |--- value: [249930.74]
|   |   |   |   |   |   |--- feature_0 >  2010.50
|   |   |   |   |   |   |   |--- value: [202131.16]
|   |   |   |   |--- feature_0 >  2011.50
|   |   |   |   |   |--- feature_2 <= 19.50
|   |   |   |   |   |   |--- feature_1 <= 26362.63
|   |   |   |   |   |   |   |--- value: [850000.00]
|   |   |   |   |   |   |--- feature_1 >  26362.63
|   |   |   |   |   |   |   |--- value: [95000.00]
|   |   |   |   |   |--- feature_2 >  19.50
|   |   |   |   |   |   |--- feature_3 <= 10.00
|   |   |   |   |   |   |   |--- feature_1 <= 26090.21
|   |   |   |   |   |   |   |   |--- value: [161130.15]
|   |   |   |   |   |   |   |--- feature_1 >  26090.21
|   |   |   |   |   |   |   |   |--- value: [182280.93]
|   |   |   |   |   |   |--- feature_3 >  10.00
|   |   |   |   |   |   |   |--- feature_3 <= 23.00
|   |   |   |   |   |   |   |   |--- value: [147814.90]
|   |   |   |   |   |   |   |--- feature_3 >  23.00
|   |   |   |   |   |   |   |   |--- value: [184126.29]
|   |   |   |--- feature_3 >  24.50
|   |   |   |   |--- feature_0 <= 2012.50
|   |   |   |   |   |--- feature_1 <= 25489.94
|   |   |   |   |   |   |--- value: [262333.33]
|   |   |   |   |   |--- feature_1 >  25489.94
|   |   |   |   |   |   |--- value: [303114.06]
|   |   |   |   |--- feature_0 >  2012.50
|   |   |   |   |   |--- feature_1 <= 25715.06
|   |   |   |   |   |   |--- value: [248778.16]
|   |   |   |   |   |--- feature_1 >  25715.06
|   |   |   |   |   |   |--- value: [263419.67]
|   |--- feature_1 >  26827.85
|   |   |--- feature_1 <= 28925.80
|   |   |   |--- feature_3 <= 24.50
|   |   |   |   |--- feature_1 <= 27131.17
|   |   |   |   |   |--- feature_1 <= 27105.85
|   |   |   |   |   |   |--- feature_0 <= 2016.00
|   |   |   |   |   |   |   |--- feature_3 <= 12.00
|   |   |   |   |   |   |   |   |--- value: [215216.11]
|   |   |   |   |   |   |   |--- feature_3 >  12.00
|   |   |   |   |   |   |   |   |--- value: [256142.06]
|   |   |   |   |   |   |--- feature_0 >  2016.00
|   |   |   |   |   |   |   |--- feature_0 <= 2017.50
|   |   |   |   |   |   |   |   |--- value: [208533.02]
|   |   |   |   |   |   |   |--- feature_0 >  2017.50
|   |   |   |   |   |   |   |   |--- value: [174957.32]
|   |   |   |   |   |--- feature_1 >  27105.85
|   |   |   |   |   |   |--- value: [265241.68]
|   |   |   |   |--- feature_1 >  27131.17
|   |   |   |   |   |--- feature_1 <= 28431.73
|   |   |   |   |   |   |--- feature_0 <= 2015.50
|   |   |   |   |   |   |   |--- feature_0 <= 2013.50
|   |   |   |   |   |   |   |   |--- value: [161364.06]
|   |   |   |   |   |   |   |--- feature_0 >  2013.50
|   |   |   |   |   |   |   |   |--- value: [127488.82]
|   |   |   |   |   |   |--- feature_0 >  2015.50
|   |   |   |   |   |   |   |--- feature_0 <= 2019.50
|   |   |   |   |   |   |   |   |--- value: [198111.35]
|   |   |   |   |   |   |   |--- feature_0 >  2019.50
|   |   |   |   |   |   |   |   |--- value: [156521.28]
|   |   |   |   |   |--- feature_1 >  28431.73
|   |   |   |   |   |   |--- feature_0 <= 2017.00
|   |   |   |   |   |   |   |--- value: [274499.90]
|   |   |   |   |   |   |--- feature_0 >  2017.00
|   |   |   |   |   |   |   |--- feature_3 <= 17.00
|   |   |   |   |   |   |   |   |--- value: [212613.21]
|   |   |   |   |   |   |   |--- feature_3 >  17.00
|   |   |   |   |   |   |   |   |--- value: [166193.48]
|   |   |   |--- feature_3 >  24.50
|   |   |   |   |--- feature_0 <= 2015.50
|   |   |   |   |   |--- value: [311479.94]
|   |   |   |   |--- feature_0 >  2015.50
|   |   |   |   |   |--- value: [295940.00]
|   |   |--- feature_1 >  28925.80
|   |   |   |--- feature_0 <= 2011.50
|   |   |   |   |--- feature_2 <= 3.50
|   |   |   |   |   |--- feature_2 <= 1.50
|   |   |   |   |   |   |--- feature_2 <= 0.50
|   |   |   |   |   |   |   |--- value: [180800.00]
|   |   |   |   |   |   |--- feature_2 >  0.50
|   |   |   |   |   |   |   |--- value: [110000.00]
|   |   |   |   |   |--- feature_2 >  1.50
|   |   |   |   |   |   |--- feature_1 <= 29306.23
|   |   |   |   |   |   |   |--- feature_2 <= 2.50
|   |   |   |   |   |   |   |   |--- value: [178195.00]
|   |   |   |   |   |   |   |--- feature_2 >  2.50
|   |   |   |   |   |   |   |   |--- value: [163333.33]
|   |   |   |   |   |   |--- feature_1 >  29306.23
|   |   |   |   |   |   |   |--- value: [225000.00]
|   |   |   |   |--- feature_2 >  3.50
|   |   |   |   |   |--- feature_2 <= 13.50
|   |   |   |   |   |   |--- feature_2 <= 5.50
|   |   |   |   |   |   |   |--- feature_2 <= 4.50
|   |   |   |   |   |   |   |   |--- value: [251989.00]
|   |   |   |   |   |   |   |--- feature_2 >  4.50
|   |   |   |   |   |   |   |   |--- value: [540666.67]
|   |   |   |   |   |   |--- feature_2 >  5.50
|   |   |   |   |   |   |   |--- feature_2 <= 6.50
|   |   |   |   |   |   |   |   |--- value: [177625.00]
|   |   |   |   |   |   |   |--- feature_2 >  6.50
|   |   |   |   |   |   |   |   |--- value: [251635.43]
|   |   |   |   |   |--- feature_2 >  13.50
|   |   |   |   |   |   |--- feature_2 <= 15.50
|   |   |   |   |   |   |   |--- feature_1 <= 29306.23
|   |   |   |   |   |   |   |   |--- value: [310000.00]
|   |   |   |   |   |   |   |--- feature_1 >  29306.23
|   |   |   |   |   |   |   |   |--- value: [499000.03]
|   |   |   |   |   |   |--- feature_2 >  15.50
|   |   |   |   |   |   |   |--- feature_0 <= 2010.50
|   |   |   |   |   |   |   |   |--- value: [315022.09]
|   |   |   |   |   |   |   |--- feature_0 >  2010.50
|   |   |   |   |   |   |   |   |--- value: [276961.54]
|   |   |   |--- feature_0 >  2011.50
|   |   |   |   |--- feature_3 <= 10.50
|   |   |   |   |   |--- feature_3 <= 4.00
|   |   |   |   |   |   |--- feature_1 <= 29472.64
|   |   |   |   |   |   |   |--- feature_0 <= 2018.50
|   |   |   |   |   |   |   |   |--- value: [217533.57]
|   |   |   |   |   |   |   |--- feature_0 >  2018.50
|   |   |   |   |   |   |   |   |--- value: [210545.24]
|   |   |   |   |   |   |--- feature_1 >  29472.64
|   |   |   |   |   |   |   |--- value: [229148.34]
|   |   |   |   |   |--- feature_3 >  4.00
|   |   |   |   |   |   |--- feature_2 <= 7.50
|   |   |   |   |   |   |   |--- feature_2 <= 4.00
|   |   |   |   |   |   |   |   |--- value: [195500.00]
|   |   |   |   |   |   |   |--- feature_2 >  4.00
|   |   |   |   |   |   |   |   |--- value: [94262.62]
|   |   |   |   |   |   |--- feature_2 >  7.50
|   |   |   |   |   |   |   |--- feature_2 <= 15.50
|   |   |   |   |   |   |   |   |--- value: [348083.33]
|   |   |   |   |   |   |   |--- feature_2 >  15.50
|   |   |   |   |   |   |   |   |--- value: [261540.46]
|   |   |   |   |--- feature_3 >  10.50
|   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |--- feature_3 <= 17.00
|   |   |   |   |   |   |   |--- value: [169514.34]
|   |   |   |   |   |   |--- feature_3 >  17.00
|   |   |   |   |   |   |   |--- value: [153685.68]
|   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |--- feature_3 <= 18.00
|   |   |   |   |   |   |   |--- value: [257954.59]
|   |   |   |   |   |   |--- feature_3 >  18.00
|   |   |   |   |   |   |   |--- value: [235299.06]
|--- feature_1 >  30842.25
|   |--- feature_1 <= 37379.52
|   |   |--- feature_0 <= 2017.50
|   |   |   |--- feature_2 <= 4.50
|   |   |   |   |--- feature_0 <= 2013.50
|   |   |   |   |   |--- feature_2 <= 1.50
|   |   |   |   |   |   |--- feature_2 <= 0.50
|   |   |   |   |   |   |   |--- value: [84500.85]
|   |   |   |   |   |   |--- feature_2 >  0.50
|   |   |   |   |   |   |   |--- value: [41250.00]
|   |   |   |   |   |--- feature_2 >  1.50
|   |   |   |   |   |   |--- feature_2 <= 2.50
|   |   |   |   |   |   |   |--- value: [136198.84]
|   |   |   |   |   |   |--- feature_2 >  2.50
|   |   |   |   |   |   |   |--- value: [149140.00]
|   |   |   |   |--- feature_0 >  2013.50
|   |   |   |   |   |--- feature_2 <= 2.50
|   |   |   |   |   |   |--- feature_1 <= 32293.84
|   |   |   |   |   |   |   |--- feature_2 <= 1.00
|   |   |   |   |   |   |   |   |--- value: [342055.71]
|   |   |   |   |   |   |   |--- feature_2 >  1.00
|   |   |   |   |   |   |   |   |--- value: [203999.91]
|   |   |   |   |   |   |--- feature_1 >  32293.84
|   |   |   |   |   |   |   |--- feature_2 <= 1.50
|   |   |   |   |   |   |   |   |--- value: [177216.67]
|   |   |   |   |   |   |   |--- feature_2 >  1.50
|   |   |   |   |   |   |   |   |--- value: [240818.25]
|   |   |   |   |   |--- feature_2 >  2.50
|   |   |   |   |   |   |--- feature_1 <= 32293.84
|   |   |   |   |   |   |   |--- feature_2 <= 3.50
|   |   |   |   |   |   |   |   |--- value: [178333.33]
|   |   |   |   |   |   |   |--- feature_2 >  3.50
|   |   |   |   |   |   |   |   |--- value: [286333.33]
|   |   |   |   |   |   |--- feature_1 >  32293.84
|   |   |   |   |   |   |   |--- feature_1 <= 34960.64
|   |   |   |   |   |   |   |   |--- value: [284527.58]
|   |   |   |   |   |   |   |--- feature_1 >  34960.64
|   |   |   |   |   |   |   |   |--- value: [300114.91]
|   |   |   |--- feature_2 >  4.50
|   |   |   |   |--- feature_1 <= 34012.67
|   |   |   |   |   |--- feature_3 <= 6.50
|   |   |   |   |   |   |--- feature_2 <= 13.50
|   |   |   |   |   |   |   |--- feature_2 <= 10.50
|   |   |   |   |   |   |   |   |--- value: [327327.65]
|   |   |   |   |   |   |   |--- feature_2 >  10.50
|   |   |   |   |   |   |   |   |--- value: [198544.54]
|   |   |   |   |   |   |--- feature_2 >  13.50
|   |   |   |   |   |   |   |--- feature_2 <= 18.50
|   |   |   |   |   |   |   |   |--- value: [430731.36]
|   |   |   |   |   |   |   |--- feature_2 >  18.50
|   |   |   |   |   |   |   |   |--- value: [337446.99]
|   |   |   |   |   |--- feature_3 >  6.50
|   |   |   |   |   |   |--- feature_3 <= 14.00
|   |   |   |   |   |   |   |--- feature_1 <= 32999.60
|   |   |   |   |   |   |   |   |--- value: [160357.53]
|   |   |   |   |   |   |   |--- feature_1 >  32999.60
|   |   |   |   |   |   |   |   |--- value: [268378.43]
|   |   |   |   |   |   |--- feature_3 >  14.00
|   |   |   |   |   |   |   |--- feature_1 <= 31108.51
|   |   |   |   |   |   |   |   |--- value: [366446.07]
|   |   |   |   |   |   |   |--- feature_1 >  31108.51
|   |   |   |   |   |   |   |   |--- value: [321350.19]
|   |   |   |   |--- feature_1 >  34012.67
|   |   |   |   |   |--- feature_2 <= 5.50
|   |   |   |   |   |   |--- feature_1 <= 34960.64
|   |   |   |   |   |   |   |--- value: [674874.98]
|   |   |   |   |   |   |--- feature_1 >  34960.64
|   |   |   |   |   |   |   |--- value: [567333.33]
|   |   |   |   |   |--- feature_2 >  5.50
|   |   |   |   |   |   |--- feature_2 <= 18.50
|   |   |   |   |   |   |   |--- feature_2 <= 14.50
|   |   |   |   |   |   |   |   |--- value: [367560.98]
|   |   |   |   |   |   |   |--- feature_2 >  14.50
|   |   |   |   |   |   |   |   |--- value: [547767.95]
|   |   |   |   |   |   |--- feature_2 >  18.50
|   |   |   |   |   |   |   |--- feature_2 <= 20.50
|   |   |   |   |   |   |   |   |--- value: [247946.64]
|   |   |   |   |   |   |   |--- feature_2 >  20.50
|   |   |   |   |   |   |   |   |--- value: [382885.72]
|   |   |--- feature_0 >  2017.50
|   |   |   |--- feature_3 <= 20.50
|   |   |   |   |--- feature_1 <= 36027.74
|   |   |   |   |   |--- feature_0 <= 2019.50
|   |   |   |   |   |   |--- feature_3 <= 5.50
|   |   |   |   |   |   |   |--- feature_0 <= 2018.50
|   |   |   |   |   |   |   |   |--- value: [240166.26]
|   |   |   |   |   |   |   |--- feature_0 >  2018.50
|   |   |   |   |   |   |   |   |--- value: [249787.77]
|   |   |   |   |   |   |--- feature_3 >  5.50
|   |   |   |   |   |   |   |--- feature_3 <= 10.00
|   |   |   |   |   |   |   |   |--- value: [291995.78]
|   |   |   |   |   |   |   |--- feature_3 >  10.00
|   |   |   |   |   |   |   |   |--- value: [243845.94]
|   |   |   |   |   |--- feature_0 >  2019.50
|   |   |   |   |   |   |--- feature_1 <= 34685.92
|   |   |   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |   |   |--- value: [274305.84]
|   |   |   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |   |   |--- value: [290560.60]
|   |   |   |   |   |   |--- feature_1 >  34685.92
|   |   |   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |   |   |--- value: [315718.78]
|   |   |   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |   |   |--- value: [324373.06]
|   |   |   |   |--- feature_1 >  36027.74
|   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |--- value: [194384.78]
|   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |--- value: [208161.18]
|   |   |   |--- feature_3 >  20.50
|   |   |   |   |--- feature_1 <= 33972.00
|   |   |   |   |   |--- feature_1 <= 32917.73
|   |   |   |   |   |   |--- feature_0 <= 2018.50
|   |   |   |   |   |   |   |--- value: [378966.19]
|   |   |   |   |   |   |--- feature_0 >  2018.50
|   |   |   |   |   |   |   |--- value: [328462.94]
|   |   |   |   |   |--- feature_1 >  32917.73
|   |   |   |   |   |   |--- value: [410186.07]
|   |   |   |   |--- feature_1 >  33972.00
|   |   |   |   |   |--- value: [482281.77]
|   |--- feature_1 >  37379.52
|   |   |--- feature_2 <= 13.50
|   |   |   |--- feature_2 <= 3.50
|   |   |   |   |--- feature_0 <= 2019.50
|   |   |   |   |   |--- feature_2 <= 2.50
|   |   |   |   |   |   |--- feature_1 <= 38154.47
|   |   |   |   |   |   |   |--- feature_2 <= 0.50
|   |   |   |   |   |   |   |   |--- value: [279000.00]
|   |   |   |   |   |   |   |--- feature_2 >  0.50
|   |   |   |   |   |   |   |   |--- value: [189609.70]
|   |   |   |   |   |   |--- feature_1 >  38154.47
|   |   |   |   |   |   |   |--- feature_2 <= 0.50
|   |   |   |   |   |   |   |   |--- value: [299500.00]
|   |   |   |   |   |   |   |--- feature_2 >  0.50
|   |   |   |   |   |   |   |   |--- value: [257453.07]
|   |   |   |   |   |--- feature_2 >  2.50
|   |   |   |   |   |   |--- feature_0 <= 2018.50
|   |   |   |   |   |   |   |--- value: [303613.00]
|   |   |   |   |   |   |--- feature_0 >  2018.50
|   |   |   |   |   |   |   |--- value: [306181.82]
|   |   |   |   |--- feature_0 >  2019.50
|   |   |   |   |   |--- feature_2 <= 0.50
|   |   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |   |--- value: [621169.75]
|   |   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |   |--- value: [325292.86]
|   |   |   |   |   |--- feature_2 >  0.50
|   |   |   |   |   |   |--- feature_2 <= 1.50
|   |   |   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |   |   |--- value: [183980.40]
|   |   |   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |   |   |--- value: [305250.00]
|   |   |   |   |   |   |--- feature_2 >  1.50
|   |   |   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |   |   |--- value: [353948.75]
|   |   |   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |   |   |--- value: [346000.00]
|   |   |   |--- feature_2 >  3.50
|   |   |   |   |--- feature_2 <= 11.50
|   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |--- feature_2 <= 6.50
|   |   |   |   |   |   |   |--- feature_2 <= 5.50
|   |   |   |   |   |   |   |   |--- value: [462691.12]
|   |   |   |   |   |   |   |--- feature_2 >  5.50
|   |   |   |   |   |   |   |   |--- value: [337078.67]
|   |   |   |   |   |   |--- feature_2 >  6.50
|   |   |   |   |   |   |   |--- feature_2 <= 9.50
|   |   |   |   |   |   |   |   |--- value: [456386.44]
|   |   |   |   |   |   |   |--- feature_2 >  9.50
|   |   |   |   |   |   |   |   |--- value: [546566.67]
|   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |--- feature_2 <= 5.50
|   |   |   |   |   |   |   |--- feature_2 <= 4.50
|   |   |   |   |   |   |   |   |--- value: [495888.94]
|   |   |   |   |   |   |   |--- feature_2 >  4.50
|   |   |   |   |   |   |   |   |--- value: [660187.50]
|   |   |   |   |   |   |--- feature_2 >  5.50
|   |   |   |   |   |   |   |--- feature_2 <= 6.50
|   |   |   |   |   |   |   |   |--- value: [430035.81]
|   |   |   |   |   |   |   |--- feature_2 >  6.50
|   |   |   |   |   |   |   |   |--- value: [495517.23]
|   |   |   |   |--- feature_2 >  11.50
|   |   |   |   |   |--- feature_2 <= 12.50
|   |   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |   |--- feature_0 <= 2018.50
|   |   |   |   |   |   |   |   |--- value: [258069.58]
|   |   |   |   |   |   |   |--- feature_0 >  2018.50
|   |   |   |   |   |   |   |   |--- value: [238642.91]
|   |   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |   |--- value: [277222.22]
|   |   |   |   |   |--- feature_2 >  12.50
|   |   |   |   |   |   |--- feature_1 <= 38154.47
|   |   |   |   |   |   |   |--- value: [350629.38]
|   |   |   |   |   |   |--- feature_1 >  38154.47
|   |   |   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |   |   |--- value: [313479.39]
|   |   |   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |   |   |--- value: [320849.95]
|   |   |--- feature_2 >  13.50
|   |   |   |--- feature_2 <= 17.50
|   |   |   |   |--- feature_2 <= 16.50
|   |   |   |   |   |--- feature_2 <= 15.50
|   |   |   |   |   |   |--- feature_2 <= 14.50
|   |   |   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |   |   |--- value: [472888.86]
|   |   |   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |   |   |--- value: [533214.25]
|   |   |   |   |   |   |--- feature_2 >  14.50
|   |   |   |   |   |   |   |--- feature_1 <= 38154.47
|   |   |   |   |   |   |   |   |--- value: [612750.00]
|   |   |   |   |   |   |   |--- feature_1 >  38154.47
|   |   |   |   |   |   |   |   |--- value: [764666.67]
|   |   |   |   |   |--- feature_2 >  15.50
|   |   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |   |--- feature_1 <= 39546.96
|   |   |   |   |   |   |   |   |--- value: [388958.33]
|   |   |   |   |   |   |   |--- feature_1 >  39546.96
|   |   |   |   |   |   |   |   |--- value: [370000.00]
|   |   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |   |--- value: [423833.33]
|   |   |   |   |--- feature_2 >  16.50
|   |   |   |   |   |--- feature_1 <= 38154.47
|   |   |   |   |   |   |--- value: [627143.88]
|   |   |   |   |   |--- feature_1 >  38154.47
|   |   |   |   |   |   |--- feature_0 <= 2019.50
|   |   |   |   |   |   |   |--- value: [893750.00]
|   |   |   |   |   |   |--- feature_0 >  2019.50
|   |   |   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |   |   |--- value: [646333.33]
|   |   |   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |   |   |--- value: [696250.00]
|   |   |   |--- feature_2 >  17.50
|   |   |   |   |--- feature_2 <= 21.50
|   |   |   |   |   |--- feature_1 <= 40856.59
|   |   |   |   |   |   |--- feature_0 <= 2019.50
|   |   |   |   |   |   |   |--- feature_2 <= 20.50
|   |   |   |   |   |   |   |   |--- value: [414866.05]
|   |   |   |   |   |   |   |--- feature_2 >  20.50
|   |   |   |   |   |   |   |   |--- value: [360925.44]
|   |   |   |   |   |   |--- feature_0 >  2019.50
|   |   |   |   |   |   |   |--- feature_2 <= 19.50
|   |   |   |   |   |   |   |   |--- value: [370338.22]
|   |   |   |   |   |   |   |--- feature_2 >  19.50
|   |   |   |   |   |   |   |   |--- value: [464236.84]
|   |   |   |   |   |--- feature_1 >  40856.59
|   |   |   |   |   |   |--- feature_2 <= 20.50
|   |   |   |   |   |   |   |--- feature_2 <= 19.50
|   |   |   |   |   |   |   |   |--- value: [337666.67]
|   |   |   |   |   |   |   |--- feature_2 >  19.50
|   |   |   |   |   |   |   |   |--- value: [329785.71]
|   |   |   |   |   |   |--- feature_2 >  20.50
|   |   |   |   |   |   |   |--- value: [392916.67]
|   |   |   |   |--- feature_2 >  21.50
|   |   |   |   |   |--- feature_1 <= 39546.96
|   |   |   |   |   |   |--- feature_1 <= 38154.47
|   |   |   |   |   |   |   |--- value: [481819.72]
|   |   |   |   |   |   |--- feature_1 >  38154.47
|   |   |   |   |   |   |   |--- value: [444728.76]
|   |   |   |   |   |--- feature_1 >  39546.96
|   |   |   |   |   |   |--- feature_0 <= 2020.50
|   |   |   |   |   |   |   |--- value: [517380.66]
|   |   |   |   |   |   |--- feature_0 >  2020.50
|   |   |   |   |   |   |   |--- value: [482350.23]

<table>
    <tr>
        <th>Metric</th>
        <th>Training</th>
        <th>Test</th>
        <th>5-Fold Cross-Validation</th>
    </tr>
    <tr>
        <td>MAE</td>
        <td>103903.89</td>
        <td>108481.67</td>
        <td>108599.59</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>23786223710.69</td>
        <td>26617011526.23</td>
        <td>26294193842.12</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>154227.83</td>
        <td>163147.21</td>
        <td>162108.93</td>
    </tr>
    <tr>
        <td>R2</td>
        <td>0.3745</td>
        <td>0.3466</td>
        <td>0.3218</td>
    </tr>
</table>

#### Training
Based on the MAE, this model is off by 103,903 euro on average, which is a small improvement from the Linear Regression model based on training data. As stated before, considering the majority of properties were less than 200,000 euro this is still a large proportion to be off by. The RMSE showed similiarly small improvement over the Linear Regression model based on training data. The R-squared score of .3745 is an improvement, but there is still room to improve.<br>

#### Test
The test data performed slightly less as well as the training data. There was a drop in the R-squared from 0.3745 to 0.3466.<br>

#### 5-Fold Cross-Validation
We can see a slight improvement overall from the past Linear Regression model CV R-squared (0.2630). Moving in the right direction, but needs more improvement.<br>

### Random Forest
<table>
    <tr>
        <th>Metric</th>
        <th>Training</th>
        <th>Test</th>
        <th>5-Fold Cross-Validation</th>
    </tr>
    <tr>
        <td>MAE</td>
        <td>99622.19</td>
        <td>108954.88</td>
        <td>107692.97</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>22232938627.59</td>
        <td>27216804263.40</td>
        <td>25932614948.26</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>149107.14</td>
        <td>164975.16</td>
        <td>160991.27</td>
    </tr>
    <tr>
        <td>R2</td>
        <td>0.4154</td>
        <td>0.3319</td>
        <td>0.3310</td>
    </tr>
</table>

#### Training

#### Test

#### 5-Fold Cross-Validation

## Conclusion/Findings

## References
- [Derelict House Image: Geograph](https://www.geograph.ie/photo/3116632)