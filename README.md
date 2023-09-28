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
    <img src='/media/data_cleaning_plan.png'>
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
        <td>117629.1663</td>
        <td>122111.0887</td>
        <td>118973.7094</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>47315291772.1562</td>
        <td>61090091876.4180</td>
        <td>51448426124.6040</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>217520.7847</td>
        <td>247164.0991</td>
        <td>226312.0957</td>
    </tr>
    <tr>
        <td>R2</td>
        <td>0.1888</td>
        <td>0.1711</td>
        <td>0.1831</td>
    </tr>
</table>

### Decision Tree
<table>
    <tr>
        <th>Metric</th>
        <th>Training</th>
        <th>Test</th>
        <th>5-Fold Cross-Validation</th>
    </tr>
    <tr>
        <td>MAE</td>
        <td>108540.7352</td>
        <td>120485.1271</td>
        <td>116307.5895</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>40003781692.9543</td>
        <td>58438504580.1186</td>
        <td>51321658145.0277</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>200009.4540</td>
        <td>241740.5729</td>
        <td>225460.8559</td>
    </tr>
    <tr>
        <td>R2</td>
        <td>0.3142</td>
        <td>0.2070</td>
        <td>0.1843</td>
    </tr>
</table>

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
        <td>104316.0195</td>
        <td>120701.0596</td>
        <td>114898.3001</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>37362284921.5016</td>
        <td>60160423267.1040</td>
        <td>49155417844.5314</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>193293.2614</td>
        <td>245276.2183</td>
        <td>221061.5852</td>
    </tr>
    <tr>
        <td>R2</td>
        <td>0.3594</td>
        <td>0.1837</td>
        <td>0.2195</td>
    </tr>
</table>

## Conclusion/Findings

## References
- Derelict House Image: Geograph - https://www.geograph.ie/photo/3116632