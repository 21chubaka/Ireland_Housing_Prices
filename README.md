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
It should be noted, that in addition to the data cleaning plan, there were some initial changes to feature names, exculsions of symbols in the price feature, and updates to data types to facilitate better data transformations and analysis.<br>

## Exploratory Analysis

## Models

## Performance

## Conclusion/Findings

## References
- Derelict House Image: Geograph - https://www.geograph.ie/photo/3116632