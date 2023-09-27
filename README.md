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

## Models

## Performance

## Conclusion/Findings

## References
- Derelict House Image: Geograph - https://www.geograph.ie/photo/3116632