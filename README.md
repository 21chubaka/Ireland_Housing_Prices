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
Property Services Regulatory Authority (PSRA) - The Residential Property Price Register (RPPR) for 2010-2021<br>
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

Central Statistics Office provided by the government of Ireland - Income Per Person and Income Indices by County by Year for 2010-2019<br>
Features:<br>
- Year:                                  int64
- County:                              string
- Income_Indices:                      float64
- Income_Per_Person_euro:              float64

## Data Cleaning
Intial data exploration was carried out to better understand the RPPR data and identify any data cleaning that needed to be carried out before
modeling.<br>

### 'Price (€)' feature
The 'Price (€)' feature contained outliers that were signficantly skewing the data, especially upper bound outliers.  The impact of the outliers
can be observed from the boxplot below.
<figure>
    <img src='/media/rppr_price_boxplot.png'>
</figure>

### 'Postal Code' feature

### 'Description of Property' feature

### 'Property Size Description' feature

### Data Cleaning Plan
<figure>
    <img src='/media/data_cleaning_plan.png'>
</figure>

## Exploratory Analysis

## Models

## Performance

## Conclusion/Findings

## References
- Derelict House Image: Geograph - https://www.geograph.ie/photo/3116632